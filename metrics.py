import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, normalized_root_mse
from typing import Tuple


def SSIM(rec, ref):
    """
    Get SSIM metrics for a reconstruction / reference pair of slices
    """
    data_range = ref.max() - ref.min()
    ssim = structural_similarity(ref, rec, data_range=data_range)

    return ssim


def pSNR(rec, ref):
    """
    Get pSNR metrics for a reconstruction / reference pair of slices
    """
    data_range = ref.max() - ref.min()
    psnr = peak_signal_noise_ratio(ref, rec, data_range=data_range)

    return psnr

def nRMSE(rec, ref):
    '''
    Get normalized roort mean squared error map for a reconstruction / reference pair of slices

    '''
    numerator = np.abs(ref - rec)
    denominator = np.mean(np.abs(ref))
    return numerator / denominator

def phase_metric(rec, ref, return_map=False):
    map = np.abs(np.angle(np.conjugate(rec) * ref))
    weights = np.abs(ref)
    if return_map:
        return map * weights / np.max(weights)
    else:
        return np.average(map, weights=weights)
    
def mae(rec, ref, return_map=False):
    map = np.abs(rec - ref)
    if return_map:
        return map 
    else:
        return np.mean(map)
