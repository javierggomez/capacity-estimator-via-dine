"""
test_mimo.py

Technische Universitaet Muenchen - Lehrstuhl fuer Nachrichtentechnik
Date: 04. März 2020
Author: Javier Garcia <javier.garcia@tum.de>
"""

import tensorflow as tf
import comm
import numpy as np
import algorithm

import tensorflow as tf

from configs import ConfigMIMO

batch_size=8000

config=ConfigMIMO()
n=config.n
P=config.P

channel=algorithm.MIMO(config.H, config.Rw, (batch_size, 1, n))

capacity=channel.capacity(P)

# apply waterfilling to obtain transformation matrix and signal levels
(CC, signal_levels, U)=comm.waterfilling_real(P, config.H, config.Rw)
V=U*np.sqrt(signal_levels[np.newaxis, :]) # transformation matrix of input

Q=V @ V.T # optimal covariance matrix of input

Ry=config.H @ Q @ config.H.conj().T+config.Rw # covariance matrix of output

# generate transmission symbols and adapt them to the eigenvalue channels (by multiplying them with V)
x_base=V @ np.random.randn(n, batch_size)
# reshape and reorder the transmit symbols to the expected shape of the channel class (batch_size, 1, n)
x=np.transpose(x_base[:, :, np.newaxis], (1, 2, 0))

# apply channel
y=channel.call(tf.convert_to_tensor(x, dtype=tf.float32)).numpy()
# reshape and reorder received symbols to shape (n, batch_size)
y_base=np.transpose(y[:, 0, :])

# estimate output entropy as -mean(log(q_Y(y)))
h_Y=np.mean(np.sum(y_base * np.linalg.solve(Ry, y_base), axis=0), axis=0)/2.0 + np.log(np.linalg.det(2.0*np.pi*Ry))/2.0
# estimate conditional entropy as -mean(log(q_{Y|X}(y|x)))
h_Y_X=np.mean(np.sum((y_base-config.H @ x_base) * np.linalg.solve(config.Rw, (y_base-config.H @ x_base)), axis=0), axis=0)/2.0 + np.log(np.linalg.det(2.0*np.pi*config.Rw))/2.0

# compute Monte-Carlo achievable rate
I_sim=h_Y-h_Y_X

print("Analytic capacity: {:.5f}".format(capacity))
print("Monte-Carlo achievable rate: {:.5f}".format(I_sim))






