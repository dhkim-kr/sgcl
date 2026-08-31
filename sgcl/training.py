"""Training loops: SGCL transductive loop (paper Algorithm 1) and the
supervised-only loops used by the method ablation (Table 8)."""

import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score

from .augmentation import (build_edge_probs, build_feature_probs,
                           uniform_drop_edges, uniform_drop_features)
from .graph import normalize_adj


def accuracy(out: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    pred = F.softmax(out, dim=1).argmax(dim=1)
    return torch.eq(pred, label).float().mean() * 100


def train_transductive(model, optimizer, feature, adj, label,
                       train_mask, test_mask, cfg, verbose: bool = False,
                       timed: bool = False) -> dict:
    """Train on the labeled nodes and evaluate every epoch on the unlabeled ones.

    Per-epoch test accuracy is the better of the two augmented views; the
    reported result is the best test accuracy over all epochs, with F1 score
    and a row-normalized confusion matrix computed from that epoch's prediction.

    With `timed=True` the cumulative wall-clock time of the training step alone
    (augmentation, forward passes, losses, backward, optimizer step — evaluation
    excluded) is measured, CUDA-synchronized, as in the paper's Table 6.
    """
    device = feature.device
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    train_time = 0.0
    eye = torch.eye(feature.shape[0], device=device)
    f1_probs = build_feature_probs(feature, cfg["pf1"])
    f2_probs = build_feature_probs(feature, cfg["pf2"])
    e1_probs = build_edge_probs(adj, cfg["pe1"])
    e2_probs = build_edge_probs(adj, cfg["pe2"])

    y_train = label[train_mask]
    y_test = label[test_mask]
    loss_lambda = cfg["loss_lambda"]

    best_acc, best_epoch = 0.0, 0
    best_pred = best_z = best_result = None

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        if timed:
            sync()
            step_start = time.time()
        optimizer.zero_grad()

        x1 = uniform_drop_features(feature, f1_probs)
        x2 = uniform_drop_features(feature, f2_probs)
        e1 = normalize_adj(uniform_drop_edges(adj, e1_probs, eye), eye)
        e2 = normalize_adj(uniform_drop_edges(adj, e2_probs, eye), eye)

        z1 = model.projection(model(x1, e1))
        z2 = model.projection(model(x2, e2))
        r1 = model.classification(z1)
        r2 = model.classification(z2)

        labeled_loss = (F.cross_entropy(r1[train_mask], y_train)
                        + F.cross_entropy(r2[train_mask], y_train)) / 2.0
        contrastive = model.contrastive_loss(z1, z2)
        loss = labeled_loss + loss_lambda * contrastive
        loss.backward()
        optimizer.step()
        if timed:
            sync()
            train_time += time.time() - step_start

        tr1_acc = accuracy(r1[test_mask], y_test)
        tr2_acc = accuracy(r2[test_mask], y_test)
        if tr1_acc > tr2_acc:
            tr_acc, tr_pred, tr_z, tr_result = tr1_acc, r1[test_mask], z1, r1
        else:
            tr_acc, tr_pred, tr_z, tr_result = tr2_acc, r2[test_mask], z2, r2

        if tr_acc > best_acc:
            best_acc, best_epoch = tr_acc, epoch
            best_pred, best_z, best_result = tr_pred, tr_z, tr_result

        if verbose and epoch % 100 == 0:
            train_acc = (accuracy(r1[train_mask], y_train)
                         + accuracy(r2[train_mask], y_train)) / 2.0
            print(f"epoch {epoch:5d}  loss {loss.item():.4f}  "
                  f"train acc {train_acc.item():6.2f}  test acc {tr_acc.item():6.2f}  "
                  f"best {best_acc.item() if torch.is_tensor(best_acc) else best_acc:6.2f}")

        if float(best_acc) == 100.0:
            break

    y_true = y_test.cpu().numpy()
    y_pred = best_pred.argmax(dim=1).detach().cpu().numpy()
    n_classes = best_result.shape[1]
    average = "binary" if n_classes == 2 else "weighted"
    return {
        "best_acc": float(best_acc),
        "best_epoch": best_epoch,
        "f1": float(f1_score(y_true, y_pred, average=average)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, normalize="true"),
        "embedding": best_z.detach().cpu().numpy(),
        "train_time": train_time,
    }


def _final_epoch_results(result, embedding, y_test, test_mask, epochs) -> dict:
    pred_logits = result[test_mask]
    acc = float(accuracy(pred_logits, y_test))
    y_true = y_test.cpu().numpy()
    y_pred = pred_logits.argmax(dim=1).detach().cpu().numpy()
    average = "binary" if result.shape[1] == 2 else "weighted"
    return {
        "best_acc": acc,
        "best_epoch": epochs,
        "f1": float(f1_score(y_true, y_pred, average=average)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, normalize="true"),
        "embedding": embedding.detach().cpu().numpy(),
        "train_time": 0.0,
    }


def train_supervised_gcn(model, optimizer, feature, adj, label, train_mask,
                         test_mask, epochs, verbose: bool = False) -> dict:
    """Supervised loop of the PCA+GCN ablation arm (G1-G3): cross-entropy only,
    fixed pre-normalized adjacency; the final epoch's test metrics are reported,
    as in the original notebooks."""
    y_train, y_test = label[train_mask], label[test_mask]
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        result, embedding = model(feature, adj)
        loss = F.cross_entropy(result[train_mask], y_train)
        loss.backward()
        optimizer.step()
        if verbose and epoch % 10 == 0:
            print(f"epoch {epoch:4d}  loss {loss.item():.4f}  "
                  f"test acc {accuracy(result[test_mask], y_test).item():6.2f}")
    return _final_epoch_results(result, embedding, y_test, test_mask, epochs)


def train_supervised_encoder(model, optimizer, feature, adj, label, train_mask,
                             test_mask, epochs, verbose: bool = False) -> dict:
    """Supervised loop of the GE ablation arm (G4-G6): the SGCL model trained
    with cross-entropy only — no augmentation, no contrastive loss; final-epoch
    test metrics are reported."""
    y_train, y_test = label[train_mask], label[test_mask]
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        embedding = model.projection(model(feature, adj))
        result = model.classification(embedding)
        loss = F.cross_entropy(result[train_mask], y_train)
        loss.backward()
        optimizer.step()
        if verbose and epoch % 10 == 0:
            print(f"epoch {epoch:4d}  loss {loss.item():.4f}  "
                  f"test acc {accuracy(result[test_mask], y_test).item():6.2f}")
    return _final_epoch_results(result, embedding, y_test, test_mask, epochs)
