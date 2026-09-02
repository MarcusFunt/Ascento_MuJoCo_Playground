"""Convenience entry point for the deterministic jump benchmark."""
from evaluation.evaluate import main


if __name__ == "__main__":
    import sys
    sys.argv[1:1] = ["--stage", "jump_flat"]
    main()
