import numpy as np
import comm

h=np.array([1.0])
r_w=np.array([1.0])
P=1.0


(C, signal_levels)=comm.waterfilling_conv_real(P, h, r_w)

