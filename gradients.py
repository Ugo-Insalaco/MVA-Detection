import numpy as np
from scipy import signal

def get_block_means(img, axis):
    s = 2
    shape = (1, s) if axis == 1 else (s, 1)
    ker = np.array([1/s for _ in range(s)]).reshape(shape)
    img = signal.convolve2d(img, ker, mode='valid')
    img = img[::s] if axis == 0 else img[:, ::s]
    return img

def get_gradients(img, axis):
    s = 2
    ker = np.array([[1, -1]])
    if axis == 1:
        ker = ker.T
    img = signal.convolve2d(img, ker, mode='valid') # 
    img = img[:, ::s] if axis==0 else img[::s]
    return img

def get_orientation(grad0, grad1):
    grad1[grad0==0] = 0
    grad0[grad0==0] = 1
    theta = np.arctan(grad1/grad0)
    return theta

def compute_gradients(img):
    b0 = get_block_means(img, 0)
    b1 = get_block_means(img, 1)
    g0 = get_gradients(b0, 0)
    g1 = get_gradients(b1, 1)
    return g0, g1
