import numpy as np
from skimage import img_as_float
from skimage import exposure


def normalize_map(s_map):
    # normalize the salience map (as done in MIT code)
    norm_s_map = (s_map - np.min(s_map))/((np.max(s_map)-np.min(s_map))*1.0)
    return norm_s_map

def normalize(x, method='standard', axis=None):
    '''Normalizes the input with specified method.
    Parameters
    ----------
    x : array-like
    method : string, optional
        Valid values for method are:
        - 'standard': mean=0, std=1
        - 'range': min=0, max=1
        - 'sum': sum=1
    axis : int, optional
        Axis perpendicular to which array is sliced and normalized.
        If None, array is flattened and normalized.
    Returns
    -------
    res : numpy.ndarray
        Normalized array.
    '''
    # TODO: Prevent divided by zero if the map is flat
    x = np.array(x, copy=False)
    if axis is not None:
        y = np.rollaxis(x, axis).reshape([x.shape[axis], -1])
        shape = np.ones(len(x.shape))
        shape[axis] = x.shape[axis]
        if method == 'standard':
            res = (x - np.mean(y, axis=1).reshape(shape)) / np.std(y, axis=1).reshape(shape)
        elif method == 'range':
            res = (x - np.min(y, axis=1).reshape(shape)) / (np.max(y, axis=1) - np.min(y, axis=1)).reshape(shape)
        elif method == 'sum':
            res = x / np.float_(np.sum(y, axis=1).reshape(shape))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    else:
        if method == 'standard':
            res = (x - np.mean(x)) / np.std(x)
        elif method == 'range':
            res = (x - np.min(x)) / (np.max(x) - np.min(x))
        elif method == 'sum':
            res = x / float(np.sum(x))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    return res


def match_hist(image, cdf, bin_centers, nbins=256):
    '''Modify pixels of input image so that its histogram matches target image histogram, specified by:
    cdf, bin_centers = cumulative_distribution(target_image)
    Parameters
    ----------
    image : array
        Image to be transformed.
    cdf : 1D array
        Values of cumulative distribution function of the target histogram.
    bin_centers ; 1D array
        Centers of bins of the target histogram.
    nbins : int, optional
        Number of bins for image histogram.
    Returns
    -------
    out : float array
        Image array after histogram matching.
    References
    ----------
    [1] Matlab implementation histoMatch(MTX, N, X) by Simoncelli, 7/96.
    '''
    image = img_as_float(image)
    old_cdf, old_bin = exposure.cumulative_distribution(image, nbins) # Unlike [1], we didn't add small positive number to the histogram
    new_bin = np.interp(old_cdf, cdf, bin_centers)
    out = np.interp(image.ravel(), old_bin, new_bin)
    return out.reshape(image.shape)
