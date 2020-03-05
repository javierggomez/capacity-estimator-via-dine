import numpy as np
import comm

eigs_noise=np.array([0.1, 0.5])
water_level=1.0
eigs_signal=water_level-eigs_noise
P=np.sum(eigs_signal)
C_calc=0.5*np.sum(np.log(1.0+eigs_signal/eigs_noise))

eigs_HHCwwinvH=1.0/eigs_noise
Cww=np.eye(2)
phi=np.random.rand(1)[0]
Qphi=np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
H=np.sqrt(eigs_HHCwwinvH)[:, np.newaxis]*Qphi


(C, signal_levels, Q)=comm.waterfilling_real(P, H, Cww)

