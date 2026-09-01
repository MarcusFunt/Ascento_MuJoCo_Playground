"""Convenience entry point for the deterministic balance benchmark."""
from evaluation.evaluate import main


if __name__ == "__main__":
    import sys
    sys.argv[1:1] = ["--stage", "balance"]
    main()
