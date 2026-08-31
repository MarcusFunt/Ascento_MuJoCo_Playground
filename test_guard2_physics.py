"""Dependency-light checks for the Guard 2.0 actuator model."""
import math
import numpy as np
from guard2_physics import Guard2ActuatorModel, LEG_Q_MIN, LEG_Q_MAX

def main():
    m=Guard2ActuatorModel(0.002)
    q=np.array([-math.pi,-math.pi,0,-math.pi,-math.pi,0.0])
    qd=np.zeros(6)
    # No command transport delay: a command affects the actuator on the same
    # physics step. Finite torque-loop bandwidth still prevents an ideal step.
    first=m.step(np.array([40,40,40,40,40,40],float),q,qd)
    assert np.max(np.abs(first)) > 0.0
    assert np.max(np.abs(first)) < 40.0
    # Wheel hard ceiling is distinct and much lower than leg ceiling.
    for _ in range(20): out=m.step(np.ones(6)*100,q,qd)
    assert abs(out[0]) <= 40.0+1e-9
    assert abs(out[2]) <= 8.0+1e-9
    # Positive torque is suppressed near the upper leg stop.
    q[0]=LEG_Q_MAX-0.01
    out=m.step(np.ones(6)*20,q,qd)
    assert out[0] <= 1e-9
    # Negative torque is suppressed near the lower stop.
    m.reset(); q[0]=LEG_Q_MIN+0.01
    for _ in range(10): out=m.step(np.array([-20,0,0,0,0,0],float),q,qd)
    assert out[0] >= -1e-9
    print('GUARD2_ACTUATOR_TEST_OK')

if __name__=='__main__': main()
