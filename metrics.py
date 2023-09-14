import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, normalized_root_mse


def SSIM(rec, ref):
    """
    Calculate SSIM metric for a reconstruction / reference pair of slices
    """
    data_range = ref.max() - ref.min()
    ssim = structural_similarity(ref, rec, data_range=data_range)

    return ssim


def PSNR(rec, ref):
    """
    Calculate PSNR metric for a reconstruction / reference pair of slices
    """
    data_range = ref.max() - ref.min()
    psnr = peak_signal_noise_ratio(ref, rec, data_range=data_range)

    return psnr


def phase_metric(rec, ref, return_map=False):
    """
    Calculate intensity-weighted absolute phase disparity metric for a 
    reconstruction / reference pair of slices. Can return the intensity-weighted
    mean, or an intensity-weighted map of values.
    """
    map = np.abs(np.angle(np.conjugate(rec) * ref))
    weights = np.abs(ref)
    if return_map:
        return map * weights / np.max(weights)
    else:
        return np.average(map, weights=weights)
    
    
def NAE(rec, ref, return_map=False):
    """
    Calculate the normalized absolute error for a reconstruction / reference 
    pair of slices. Metric is normalized by mean of reference. Can return the 
    mean NAE, or a map of values.
    """
    map = np.abs(np.abs(rec) - np.abs(ref)) / np.mean(np.abs(ref)) * 10
    if return_map:
        return map 
    else:
        return np.mean(map)
