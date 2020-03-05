import numpy as np

def eig(*args):
    """Equivalent to numpy.linalg.eig, but also works for size-1 arrays"""
    if len(args)==1 and np.size(args)==1:
        return (args[0], np.array([1]))
    else:
        return np.linalg.eig(*args)


def sqrtm(M):
    """Computes a matrix A such that A @ A.conj().T == M"""
    (L, Q)=eig(M)
    return Q @ np.diag(np.sqrt(L))


def real_correlated_gaussian(C, N):
    """Generates an array W of real Gaussian samples with shape (D, N), where the covariance matrix of the columns  of W
    is C."""
    D=np.shape(C)[0]
    if D==1:
        return np.sqrt(C) * np.random.randn(D, N)
    else:
        return sqrtm(C) @ np.random.randn(D, N)


def waterfilling_base(P, L):
    """Applies the waterfilling algorithm to a set of noise variances of independent channels. In other words, maximizes
     C=np.sum(np.log(1+S/L)), where L is a given vector and S is a vector to be found, such that np.sum(S)<=P. Returns
     (C, S)"""
    idx_sort = np.argsort(L)

    L = L[idx_sort]
    # Q = Q[:, idx_sort]

    K_orig = np.size(L)
    water_levels = (P + np.cumsum(L)) / (np.arange(1, K_orig + 1))
    K = np.argwhere(water_levels - L > 0)[-1][0] + 1

    signal_levels=np.zeros((K_orig,))
    signal_levels[:K] = water_levels[K - 1] - L[:K]

    C = np.sum(np.log(1.0 + signal_levels[:K] / L[:K]))

    signal_levelstemp=np.copy(signal_levels)
    signal_levels[idx_sort]=signal_levelstemp

    return (C, signal_levels)

def waterfilling(P, H, Rww):
    """Applies the waterfilling algorithm to a complex MIMO channel, given by the channel matrix H and the noise
      covariance matrix Rww. For this, applies an eigenvalue decomposition to H.conj().T @ inv(Rww) @ H, and applies the
      waterfilling_base algorithm to the inverse of the resulting eigenvalues. Returns (C, S, Q), where C is the
      capacity, S are the power levels corresponding to the eigenvalues, and Q is the unitary matrix resulting from the
      eigenvalue decomposition."""
    R=H.conj().T @ np.linalg.solve(Rww, H)
    (L, Q)=eig(R)
    L=1.0/L

    (C, signal_levels)=waterfilling_base(P, L)

    return (C, signal_levels, Q)

def waterfilling_real(P, H, Rww):
    """Applies the waterfilling algorithm to a real MIMO channel, given by the channel matrix H and the noise
      covariance matrix Rww. For this, applies an eigenvalue decomposition to H.conj().T @ inv(Rww) @ H, and applies the
      waterfilling_base algorithm to the inverse of the resulting eigenvalues. Returns (C, S, Q), where C is the
      capacity, S are the power levels corresponding to the eigenvalues, and Q is the unitary matrix resulting from the
      eigenvalue decomposition."""
    R=H.conj().T @ np.linalg.solve(Rww, H)
    (L, Q)=eig(R)
    L=1.0/L

    (C, signal_levels)=waterfilling_base(P, L)

    return (C/2.0, signal_levels, Q)

def waterfilling_conv(P, h, r_w):
    """Applies the waterfilling algorithm to a complex convolutional channel, given by the channel impulse response h and the noise
      auto-correlation function r_w. For this, transforms the channel to a set of parallel, independent channels by taking FFT of h and r_w, and applies the
      waterfilling_base algorithm to the resulting R_w/np.abs(H)**2. Returns (C, S), where C is the
      capacity, and S are the power levels corresponding to the frequency components."""
    N_fft=128
    n_r_w=np.size(r_w)
    hh=np.fft.fft(h, N_fft)
    PSD_w=np.fft.fft(np.concatenate((r_w, np.zeros((N_fft-(2*n_r_w-1),)), r_w[-1:0:-1]), axis=0), N_fft)
    L=PSD_w / np.abs(hh)**2
    # (C, signal_levels, Q)=waterfilling(P*N_fft, np.diag(hh), np.diag(PSD_w))
    (C, signal_levels)=waterfilling_base(P*N_fft, L)
    return (C/N_fft, signal_levels)

def waterfilling_conv_real(P, h, r_w):
    """Applies the waterfilling algorithm to a real convolutional channel, given by the channel impulse response h and the noise
      auto-correlation function r_w. For this, transforms the channel to a set of parallel, independent channels by taking FFT of h and r_w, and applies the
      waterfilling_base algorithm to the resulting R_w/np.abs(H)**2. Returns (C, S), where C is the
      capacity, and S are the power levels corresponding to the frequency components."""
    N_fft=128
    n_r_w=np.size(r_w)
    hh=np.fft.fft(h, N_fft)
    PSD_w=np.fft.fft(np.concatenate((r_w, np.zeros((N_fft-(2*n_r_w-1),)), r_w[-1:0:-1]), axis=0), N_fft)
    assert np.all(np.isreal(PSD_w))
    PSD_w=np.real(PSD_w)
    L = PSD_w / np.abs(hh) ** 2
    (C, signal_levels)=waterfilling_base(P*N_fft, L)
    return (C/N_fft/2.0, signal_levels)



