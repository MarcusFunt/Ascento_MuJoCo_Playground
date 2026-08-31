"""Compatibility entry point for the real Brax PPO balance trainer.
No teacher data, PD initialization, or imitation loss is used.
"""
from training.train import main

if __name__ == "__main__":
    main()
