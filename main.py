"""SGCL entry point.

Examples:
    python main.py --config configs/seed.yaml
    python main.py --config configs/deap_valence.yaml --cases 3 --device cuda:1
    python main.py --config configs/seed_iv.yaml --subjects 1 2 3 --verbose
"""

import argparse

import torch
import yaml

from sgcl.ablation import SCHEMES
from sgcl.experiment import run


def parse_args():
    parser = argparse.ArgumentParser(description="Semi-supervised Graph Contrastive "
                                                 "Learning for EEG emotion recognition")
    parser.add_argument("--config", required=True, help="path to a YAML config file")
    parser.add_argument("--device", help="override experiment.device (e.g. cuda:1, cpu)")
    parser.add_argument("--cases", type=int, nargs="+",
                        help="override experiment.run_cases (e.g. --cases 3)")
    parser.add_argument("--subjects", type=int, nargs="+",
                        help="1-based subject numbers to run (default: all)")
    parser.add_argument("--epochs", type=int, help="override train.epochs")
    parser.add_argument("--output-dir", help="override experiment.output_dir")
    parser.add_argument("--verbose", action="store_true",
                        help="print per-epoch training progress")
    parser.add_argument("--mode", choices=["main", "ablation", "grid", "saliency",
                                           "tsne"],
                        help="main results (default), method ablation G1-G9 "
                             "(Tables 7-8), pe/pf sensitivity grid (Fig. 6), "
                             "saliency topomaps (Fig. 5), or t-SNE scatter (Fig. 4)")
    parser.add_argument("--schemes", nargs="+", choices=list(SCHEMES),
                        help="ablation schemes to run (default: all nine)")
    parser.add_argument("--graph-variant",
                        choices=["full", "no_knn", "no_broad_knn", "no_local_knn"],
                        help="KNN-necessity ablation variant (Table 10)")
    parser.add_argument("--timed", action="store_true",
                        help="measure the CUDA-synchronized training-step time "
                             "(Table 6)")
    parser.add_argument("--save-embeddings", action="store_true",
                        help="save each subject's best node embedding (needed "
                             "for --mode tsne)")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or not {"dataset", "experiment", "train"} <= set(cfg):
        raise SystemExit(f"{args.config} is not a valid SGCL config "
                         "(needs dataset/experiment/train sections)")

    if args.device is not None:
        torch.device(args.device)  # fail fast on malformed device strings
        cfg["experiment"]["device"] = args.device
    if args.cases:
        cfg["experiment"]["run_cases"] = args.cases
    if args.subjects:
        if min(args.subjects) < 1:
            raise SystemExit("--subjects uses 1-based subject numbers")
        cfg["experiment"]["subjects"] = args.subjects
    if args.epochs is not None:
        # one budget override for every mode's training schedule
        cfg["train"]["epochs"] = args.epochs
        cfg["experiment"]["saliency_epochs"] = args.epochs
        cfg.setdefault("ablation", {})["epochs"] = args.epochs
    if args.output_dir is not None:
        cfg["experiment"]["output_dir"] = args.output_dir
    if args.verbose:
        cfg["experiment"]["verbose"] = True
    if args.mode:
        cfg["experiment"]["mode"] = args.mode
    if args.schemes:
        if cfg["experiment"].get("mode", "main") != "ablation":
            raise SystemExit("--schemes only applies with --mode ablation")
        cfg["experiment"]["schemes"] = args.schemes
    if args.graph_variant:
        cfg["graph"]["variant"] = args.graph_variant
    if args.timed:
        cfg["train"]["timed"] = True
    if args.save_embeddings:
        cfg["experiment"]["save_embeddings"] = True

    run(cfg)


if __name__ == "__main__":
    main()
