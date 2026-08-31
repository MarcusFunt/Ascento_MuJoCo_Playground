# Project goals and assumptions

## Primary goal

This MuJoCo/Ascento project exists to generate **physically plausible, visually convincing 3D robot motion** that can be used as an animation source for a 2D pixel-art game pipeline. The learned/inference controller is **simulation-only**. It will not be deployed on a physical Ascento Guard 2.0 or other robot.

The 3D simulation is therefore a motion-generation and animation-authoring system, not a sim-to-real robotics deployment stack.

## Intended animation pipeline

1. Use the Guard 2.0-like rigid-body model in MuJoCo as the physical source of motion.
2. Train or optimize policies for stable balancing, turning, crouching, jumping, flight attitude, landing, and recovery.
3. Export useful 3D trajectories/poses.
4. Render/bake those motions through the 3D-to-pixel-art pipeline.
5. Use the resulting sprite animations and metadata in the 2D game.

The final game does not need to run MuJoCo or the inference policy.

## What should be physically realistic

The model should retain dynamics that materially affect the *shape and plausibility of motion*:

- Guard 2.0-like mass distribution and inertia.
- Correct wheel and linkage geometry.
- Finite hip/knee travel.
- Distinct wheel and leg actuator capabilities.
- Torque saturation and torque-speed limits.
- Finite actuator torque response/bandwidth where it changes motion.
- Wheel/ground friction and compliant tyre contact.
- Contact, takeoff, flight, impact, and landing dynamics.
- No passive leg spring: Guard 2.0 does not have one.

## What does NOT need to be simulated for deployment realism

Unless a future experiment explicitly asks for them, do not add effects whose main purpose is sim-to-real transfer or hardware robustness:

- communication/CAN transport delay;
- artificial policy-to-actuator command latency;
- sensor noise;
- state-estimator drift;
- packet loss;
- embedded compute limits;
- onboard inference latency;
- battery/thermal limits that do not materially affect the short animation motion being studied.

The controller may use exact MuJoCo state, contact information, center-of-mass state, or other privileged simulator quantities if they improve motion quality.

## Policy assumption

The policy should be optimized for **simulation performance and motion quality**, not hardware deployability. It may therefore use richer observations, larger networks, trajectory optimization, MPC, phase-conditioned control, privileged state, or other methods that would be inconvenient on a real robot.

Robustness is still useful when it improves animation quality—for example, recovery from perturbations, stable landings, or motion across different speeds—but domain randomization should not be added merely as a sim-to-real ritual.

## Current uncertain model parameters

Some exact Guard 2.0 actuator details are not publicly documented. Values that remain estimates are centralized in `guard2_physics.py` rather than hidden throughout the project. At present the main uncertain values are:

- wheel peak and continuous torque;
- leg/wheel torque-loop time constants;
- exact hard mechanical joint stops if the current wrapped limits differ from production Guard 2.0.

These should be refined when better evidence becomes available, but the project should not invent hardware-only effects just to make the simulator look more realistic.

## Design rule

**Model the mechanics that shape the motion; omit deployment artifacts that do not.**
