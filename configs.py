import logging
import numpy as np
logger = logging.getLogger("logger")

class BasicConfig(object):
    ########## general ##########
    name = 'debug'

    ######## randomness #########
    seed = 8564456

    ######## environment ########
    config_name = "awgn"        # channel model
    n = 1                       # dimension of X,Y
    P = 1.0                     # power of X
    N = 1.0                     # power of channel's inner innovation process
    feedback = False            # compute feedback capacity
    C = None                    # capacity of the used configuration (for visualization)
    m = 1                       # dimension of NDT generator


    ######## DINE/NDT models #########
    T = 20                      # unroll of lstm models
    DI_hidden = 100             # hidden units in DINE model
    DI_last_hidden = 100        # hidden units of last layer in DINE model
    DI_dropout = 0.0            # dropout and recurrent dropout in DINE model
    NDT_hidden = 100            # hidden units in NDT model
    NDT_last_hidden = 100       # hidden units of last layer in NDT model
    NDT_dropout = 0.0           # dropout and recurrent dropout in NDT model

    ######## training #########
    opt = "adam"                # optimizer name
    lr_rate_DI = 0.0001         # lr for DINE model
    lr_rate_enc = 0.0001        # lr for NDT model
    clip_norm = 0.1             # clip norm of gradients
    batch_size = 50             # batch size for training
    epochs_di = 500             # num of epochs for initial training of DINE model
    epochs_enc = 5000           # num of epochs for interchangeably training DINE and NDT
    n_steps_mc = 250            # length of MC evaluation (total samples = batch_size * n_steps_mc * T)

    def show(self):
        attrs = [attr for attr in dir(self) if (not attr.startswith('__') and attr != "show")]
        logger.info('\n'.join("%s: %s" % (item, getattr(self, item)) for item in attrs))


class ConfigAWGN(BasicConfig):
    config_name = "awgn"        # channel model
    n = 1                       # dimension of X,Y
    P = 1.0                     # power of X
    N = 1.0                     # power of channel's inner innovation process
    feedback = False            # compute feedback capacity
    C = 0.3466                  # capacity of the used configuration (for visualization)
    m = 1                       # dimension of NDT generator

class ConfigFF_MA_AGN(BasicConfig):
    config_name = "arma_ff"     # channel model
    channel_alpha = 0.5         # channel parameter
    n = 1                       # dimension of X,Y
    P = 1.0                     # power of X
    N = 1.0                     # power of channel's inner innovation process
    feedback = False            # compute feedback capacity
    C = None                    # capacity of the used configuration (for visualization)
    m = 1                       # dimension of NDT generator

class ConfigFB_MA_AGN(BasicConfig):
    config_name = "arma_fb"     # channel model
    channel_alpha = 0.5         # channel parameter
    n = 1                       # dimension of X,Y
    P = 1.0                     # power of X
    N = 1.0                     # power of channel's inner innovation process
    feedback = True             # compute feedback capacity
    C = None                    # capacity of the used configuration (for visualization)
    m = 1                       # dimension of NDT generator

    ######## training #########
    opt = "adam"                 # optimizer name
    lr_rate_DI = 0.0001          # lr for DINE model
    lr_rate_enc = 0.00005        # lr for NDT model

class ConfigMIMO(BasicConfig):
    config_name = "mimo"  # channel model
    H=np.array([[1.0, 0.0], [0.0, 1.0]])  # channel matrix
    Rw=np.eye(2) # covariance matrix of noise
    n = 2  # dimension of X,Y
    P = 2.0  # power of X
    # N = 1.0  # power of channel's inner innovation process
    feedback = False  # compute feedback capacity
    C = None  # capacity of the used configuration (for visualization)
    m = 1  # dimension of NDT generator

class ConfigMIMO2(BasicConfig):
    config_name = "mimo"  # channel model
    H=np.array([[1.0, 0.5], [-1.0, 1.0]])  # channel matrix
    Rw=np.array([[1.0, 0.2], [0.2, 1.0]]) # covariance matrix of noise
    n = 2  # dimension of X,Y
    P = 2.0  # power of X
    # N = 1.0  # power of channel's inner innovation process
    feedback = False  # compute feedback capacity
    C = None  # capacity of the used configuration (for visualization)
    m = 1  # dimension of NDT generator