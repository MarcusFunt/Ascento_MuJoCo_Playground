"""Convenience entry point for the deterministic recovery benchmark."""
from evaluation.evaluate import main


if __name__ == "__main__":
    import sys
    sys.argv[1:1] = ["--stage", "recovery"]
    main()
