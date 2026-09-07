"""Print deterministic CPU checks for reward geometry and sparse-event scaling."""

from __future__ import annotations

import math

import torch

from ascento_mjlab.geometry import projected_gravity_tilt


def main() -> None:
    print("orientation  tilt_radians  upright_score")
    for degrees in (0, 15, 30, 60, 90, 180):
        radians = math.radians(degrees)
        gravity = torch.tensor([[math.sin(radians), 0.0, -math.cos(radians)]])
        tilt = projected_gravity_tilt(gravity).item()
        score = math.exp(-((tilt / 0.35) ** 2))
        print(f"{degrees:>3} deg       {tilt:>8.5f}      {score:.6f}")

    print("\njump event  dt=0.005  dt=0.010  dt=0.020")
    for name, weight in (("takeoff", 2.0), ("landing", 2.0), ("distance", 3.0)):
        values = "  ".join(f"{weight:.1f}" for _ in (0.005, 0.01, 0.02))
        print(f"{name:<10}  {values}")


if __name__ == "__main__":
    main()
