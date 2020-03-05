import numpy as np
import comm

alpha=0.5
sigma_u_2=1.0
h=np.array([1.0])
r_w=sigma_u_2*np.array([1.0+alpha**2, alpha])
P=1.0


(C, signal_levels)=comm.waterfilling_conv_real(P, h, r_w)

