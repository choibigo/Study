import h5py
import numpy as np
import os

def load(scale_factor=2):
    root = os.path.dirname(os.path.abspath(__file__))
    f = h5py.File(os.path.join(root, f"SRMDNFx{scale_factor}.mat"), 'r')
    
    directKernel = None
    if scale_factor != 4:
        directKernel = f['net/meta/directKernel']
        directKernel = np.array(directKernel).transpose(3, 2, 1, 0)

    AtrpGaussianKernels = f['net/meta/AtrpGaussianKernel']
    AtrpGaussianKernels = np.array(AtrpGaussianKernels).transpose(3, 2, 1, 0)

    P = f['net/meta/P']
    P = np.array(P)
    P = P.T

    if directKernel is not None:
        K = np.concatenate((directKernel, AtrpGaussianKernels), axis=-1)
    else:
        K = AtrpGaussianKernels

    return K, P