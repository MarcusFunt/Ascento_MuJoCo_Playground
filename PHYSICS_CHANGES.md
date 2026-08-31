# Guard 2.0 physics patch — simulation-only revision

This patch deliberately leaves the supplied CAD masses and inertia tensors unchanged.

## Implemented

- Finite hip/knee travel in the simulation (`-1.5*pi .. -0.5*pi` in this URDF's wrapped joint convention), plus a 5 degree actuator soft-guard zone.
- Separate leg and wheel actuator profiles. Wheels no longer inherit the leg's ±40 Nm ideal torque source.
- First-order actuator torque response, hard torque saturation, torque-speed derating, and velocity protection.
- Rubber/pneumatic-style wheel contact: `mu=0.8`, torsional/rolling friction, 6D contact, and compliant MuJoCo `solref/solimp` settings.
- **No passive leg spring or joint stiffness. Guard 2.0 has no physical spring.**
- **No artificial actuator command/transport delay.** The inference policy is simulation-only and will never be deployed on physical hardware.

## Why command delay was removed

The previous revision inserted millisecond-scale command transport delays as a sim-to-real precaution. That is not part of this project's goal. The policy can act directly on the simulated plant, so communication/CAN/controller-stack latency would only make optimization harder without improving the intended animation output.

Finite torque response is still retained because it represents actuator/mechanical dynamics and materially affects acceleration, balance, takeoff impulse, and landing behavior.

## Parameters that are evidence-backed vs estimates

Evidence-backed / inherited from the supplied Ascento model:
- leg hard effort envelope: 40 Nm
- leg controller velocity envelope: 4 rad/s
- wheel URDF velocity envelope: 20 rad/s
- ROS wheel command interface: ±10 rad/s
- wheel radius and rigid-body inertias/masses: unchanged

Conservative estimates pending better Guard 2.0 actuator characterization:
- wheel peak/continuous torque: 8 / 5 Nm
- torque-loop time constants: 4 ms legs, 3 ms wheels

All uncertain actuator values live in `guard2_physics.py`.

## Important

The existing balance policy was trained against ideal ±40 Nm wheel motors. It should be retrained after this patch because wheel authority, torque-speed limits, finite actuator response, joint stops, and compliant tyre contact change the plant. No retraining is needed merely because command transport delay was removed; this revision makes the plant more appropriate for simulation-only motion generation.
