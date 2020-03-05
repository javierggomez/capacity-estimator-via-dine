"""
test_mimo.py

Technische Universitaet Muenchen - Lehrstuhl fuer Nachrichtentechnik
Date: 04. März 2020
Author: Javier Garcia <javier.garcia@tum.de>
(C) All rights reserved
"""

import tensorflow as tf
import comm
import numpy as np
import algorithm

import tensorflow as tf

from configs import ConfigMIMO2

batch_size=2000

config=ConfigMIMO2()
n=config.n
P=config.P

channel=algorithm.MIMO(config.H, config.Rw, (batch_size, 1, n))

capacity=channel.capacity(P)

(CC, signal_levels, U)=comm.waterfilling_real(P, config.H, config.Rw)
V=U*np.sqrt(signal_levels[np.newaxis, :]) # transformation matrix of input

Q=V @ V.T # optimal covariance matrix of input

Ry=config.H @ Q @ config.H.conj().T+config.Rw # covariance matrix of output


x_base=V @ np.random.randn(n, batch_size)
x=np.transpose(x_base[:, :, np.newaxis], (1, 2, 0))
y=channel.call(tf.convert_to_tensor(x, dtype=tf.float32)).numpy()

y_base=np.transpose(y[:, 0, :])

h_Y=np.mean(np.sum(y_base * np.linalg.solve(Ry, y_base), axis=0), axis=0)/2.0 + np.log(np.linalg.det(2.0*np.pi*Ry))/2.0
h_Y_X=np.mean(np.sum((y_base-config.H @ x_base) * np.linalg.solve(config.Rw, (y_base-config.H @ x_base)), axis=0), axis=0)/2.0 + np.log(np.linalg.det(2.0*np.pi*config.Rw))/2.0

I_sim=h_Y-h_Y_X






