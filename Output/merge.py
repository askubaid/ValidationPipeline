from pathlib import Path
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Merge specific layer2 files into learner.en and learner.pt")
    parser.add_argument("--layer2", default="../layer2",
                        help="relative path from this script to the layer2 folder (default: ../layer2)")
    parser.add_argument("--only", choices=["en", "pt", "all"], default="all",
                        help="only merge one language, or both (default: all)")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    layer2_dir = (base_dir / args.layer2).resolve()
    if not layer2_dir.exists():
        print(f"Error: layer2 directory not found at {layer2_dir}")
        sys.exit(2)

    groups = {
        "learner.en": [
            "learner_excellent.en",
            "learner_single____excellent.en",
            "learner_single__excellent.en",
        ],
        "learner.pt": [
            "learner_excellent.pt",
            "learner_single____excellent.pt",
            "learner_single__excellent.pt",
        ],
    }

    def merge_group(output_filename, filenames):
        out_path = base_dir / output_filename
        written = 0
        with out_path.open("w", encoding="utf-8") as outf:
            for fname in filenames:
                src = layer2_dir / fname
                if not src.exists():
                    print(f"Warning: {src} not found — skipping.")
                    continue
                with src.open("r", encoding="utf-8") as inf:
                    for line in inf:
                        outf.write(line)
                        written += 1
        print(f"Merged {written} lines into {out_path}")

    if args.only in ("all", "en"):
        merge_group("learner.en", groups["learner.en"])
    if args.only in ("all", "pt"):
        merge_group("learner.pt", groups["learner.pt"])


if __name__ == "__main__":
    main()
