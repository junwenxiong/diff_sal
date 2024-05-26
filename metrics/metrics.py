from functools import partial
import numpy as np
from numpy import random
from skimage.transform import resize
from .utils import normalize

def AUC_Judd(saliency_map, fixation_map, jitter=True):
    '''
    AUC stands for Area Under ROC Curve.
    This measures how well the saliency map of an image predicts the ground truth human fixations on the image.
    ROC curve is created by sweeping through threshold values
    determined by range of saliency map values at fixation locations.
    True positive (tp) rate correspond to the ratio of saliency map values above threshold
    at fixation locations to the total number of fixation locations.
    False positive (fp) rate correspond to the ratio of saliency map values above threshold
    at all other locations to the total number of possible other locations (non-fixated image pixels).
    AUC=0.5 is chance level.
    Parameters
    ----------
    saliency_map : real-valued matrix
    fixation_map : binary matrix
        Human fixation map.
    jitter : boolean, optional
        If True (default), a small random number would be added to each pixel of the saliency map.
        Jitter saliency maps that come from saliency models that have a lot of zero values.
        If the saliency map is made with a Gaussian then it does not need to be jittered
        as the values vary and there is not a large patch of the same value.
        In fact, jittering breaks the ordering in the small values!
    Returns
    -------
    AUC : float, between [0,1]
    '''
    saliency_map = np.array(saliency_map, copy=False)
    fixation_map = np.array(fixation_map, copy=False) > 0.5

    # If there are no fixation to predict, return NaN
    if not np.any(fixation_map):
        print('no fixation to predict')
        return np.nan
    # Make the saliency_map the size of the fixation_map
    if saliency_map.shape != fixation_map.shape:
        saliency_map = resize(saliency_map, fixation_map.shape, order=3, mode='reflect')
    # Jitter the saliency map slightly to disrupt ties of the same saliency value
    if jitter:
        saliency_map += random.rand(*saliency_map.shape) * 1e-7
    # Normalize saliency map to have values between [0,1]
    saliency_map = normalize(saliency_map, method='range')

    S = saliency_map.ravel()
    F = fixation_map.ravel()
    S_fix = S[F] # Saliency map values at fixation locations
    n_fix = len(S_fix)
    n_pixels = len(S)
    # Calculate AUC
    thresholds = sorted(S_fix, reverse=True)
    tp = np.zeros(len(thresholds)+2)
    fp = np.zeros(len(thresholds)+2)
    tp[0] = 0; tp[-1] = 1
    fp[0] = 0; fp[-1] = 1
    for k, thresh in enumerate(thresholds):
        above_th = np.sum(S >= thresh) # Total number of saliency map values above threshold
        tp[k+1] = (k + 1) / float(n_fix) # Ratio saliency map values at fixation locations above threshold
        fp[k+1] = (above_th - k - 1) / float(n_pixels - n_fix) # Ratio other saliency map values above threshold
    return np.trapz(tp, fp) # y, x


def AUC_Borji(saliency_map, fixation_map, n_rep=100, step_size=0.1, rand_sampler=None):
    '''
    This measures how well the saliency map of an image predicts the ground truth human fixations on the image.
    ROC curve created by sweeping through threshold values at fixed step size
    until the maximum saliency map value.
    True positive (tp) rate correspond to the ratio of saliency map values above threshold
    at fixation locations to the total number of fixation locations.
    False positive (fp) rate correspond to the ratio of saliency map values above threshold
    at random locations to the total number of random locations
    (as many random locations as fixations, sampled uniformly from fixation_map ALL IMAGE PIXELS),
    averaging over n_rep number of selections of random locations.
    Parameters
    ----------
    saliency_map : real-valued matrix
    fixation_map : binary matrix
        Human fixation map.
    n_rep : int, optional
        Number of repeats for random sampling of non-fixated locations.
    step_size : int, optional
        Step size for sweeping through saliency map.
    rand_sampler : callable
        S_rand = rand_sampler(S, F, n_rep, n_fix)
        Sample the saliency map at random locations to estimate false positive.
        Return the sampled saliency values, S_rand.shape=(n_fix,n_rep)
    Returns
    -------
    AUC : float, between [0,1]
    '''
    saliency_map = np.array(saliency_map, copy=False)
    fixation_map = np.array(fixation_map, copy=False) > 0.5
    # If there are no fixation to predict, return NaN
    if not np.any(fixation_map):
        print('no fixation to predict')
        return np.nan
    # Make the saliency_map the size of the fixation_map
    if saliency_map.shape != fixation_map.shape:
        saliency_map = resize(saliency_map, fixation_map.shape, order=3, mode='reflect')
    # Normalize saliency map to have values between [0,1]
    saliency_map = normalize(saliency_map, method='range')

    S = saliency_map.ravel()
    F = fixation_map.ravel()
    S_fix = S[F] # Saliency map values at fixation locations
    n_fix = len(S_fix)
    n_pixels = len(S)
    # For each fixation, sample n_rep values from anywhere on the saliency map
    if rand_sampler is None:
        r = random.randint(0, n_pixels, [n_fix, n_rep])
        S_rand = S[r] # Saliency map values at random locations (including fixated locations!? underestimated)
    else:
        S_rand = rand_sampler(S, F, n_rep, n_fix)
    # Calculate AUC per random split (set of random locations)
    auc = np.zeros(n_rep) * np.nan
    for rep in range(n_rep):
        thresholds = np.r_[0:np.max(np.r_[S_fix, S_rand[:,rep]]):step_size][::-1]
        tp = np.zeros(len(thresholds)+2)
        fp = np.zeros(len(thresholds)+2)
        tp[0] = 0; tp[-1] = 1
        fp[0] = 0; fp[-1] = 1
        for k, thresh in enumerate(thresholds):
            tp[k+1] = np.sum(S_fix >= thresh) / float(n_fix)
            fp[k+1] = np.sum(S_rand[:,rep] >= thresh) / float(n_fix)
        auc[rep] = np.trapz(tp, fp)
    return np.mean(auc) # Average across random splits


def AUC_shuffled(saliency_map, fixation_map, other_map, n_rep=100, step_size=0.1):
    '''
    This measures how well the saliency map of an image predicts the ground truth human fixations on the image.
    ROC curve created by sweeping through threshold values at fixed step size
    until the maximum saliency map value.
    True positive (tp) rate correspond to the ratio of saliency map values above threshold
    at fixation locations to the total number of fixation locations.
    False positive (fp) rate correspond to the ratio of saliency map values above threshold
    at random locations to the total number of random locations
    (as many random locations as fixations, sampled uniformly from fixation_map ON OTHER IMAGES),
    averaging over n_rep number of selections of random locations.
    Parameters
    ----------
    saliency_map : real-valued matrix
    fixation_map : binary matrix
        Human fixation map.
    other_map : binary matrix, same shape as fixation_map
        A binary fixation map (like fixation_map) by taking the union of fixations from M other random images
        (Borji uses M=10).
    n_rep : int, optional
        Number of repeats for random sampling of non-fixated locations.
    step_size : int, optional
        Step size for sweeping through saliency map.
    Returns
    -------
    AUC : float, between [0,1]
    '''
    other_map = np.array(other_map, copy=False) > 0.5
    #fixation_map = np.array(fixation_map, copy=False) > 0.5
    #print(other_map.dtype, fixation_map.dtype)
    if other_map.shape != fixation_map.shape:
        raise ValueError('other_map.shape != fixation_map.shape')

    # For each fixation, sample n_rep values (from fixated locations on other_map) on the saliency map
    def sample_other(other, S, F, n_rep, n_fix):
        fixated = np.nonzero(other)[0]
        indexer = map(lambda x: random.permutation(x)[:n_fix], np.tile(range(len(fixated)), [n_rep, 1]))
        #print(indexer, np.transpose(indexer).astype(np.int64))
        indexer = np.asarray(list(indexer))
        r = fixated[np.transpose(indexer)]
        S_rand = S[r] # Saliency map values at random locations (including fixated locations!? underestimated)
        return S_rand
    return AUC_Borji(saliency_map, fixation_map, n_rep, step_size, partial(sample_other, other_map.ravel()))


def NSS(saliency_map, fixation_map):
    '''
    Normalized scanpath saliency of a saliency map,
    defined as the mean value of normalized (i.e., standardized) saliency map at fixation locations.
    You can think of it as a z-score. (Larger value implies better performance.)
    Parameters
    ----------
    saliency_map : real-valued matrix
        If the two maps are different in shape, saliency_map will be resized to match fixation_map..
    fixation_map : binary matrix
        Human fixation map (1 for fixated location, 0 for elsewhere).
    Returns
    -------
    NSS : float, positive
    '''
    s_map = np.array(saliency_map, copy=False)
    f_map = np.array(fixation_map, copy=False) > 0.5
    if s_map.shape != f_map.shape:
        s_map = resize(s_map, f_map.shape)
    # Normalize saliency map to have zero mean and unit std
    s_map = normalize(s_map, method='standard')
    # Mean saliency value at fixation locations
    return np.mean(s_map[f_map])


def CC(saliency_map1, saliency_map2):
    '''
    Pearson's correlation coefficient between two different saliency maps
    (CC=0 for uncorrelated maps, CC=1 for perfect linear correlation).
    Parameters
    ----------
    saliency_map1 : real-valued matrix
        If the two maps are different in shape, saliency_map1 will be resized to match saliency_map2.
    saliency_map2 : real-valued matrix
    Returns
    -------
    CC : float, between [-1,1]
    '''
    map1 = np.array(saliency_map1, copy=False)
    map2 = np.array(saliency_map2, copy=False)
    if map1.shape != map2.shape:
        map1 = resize(map1, map2.shape, order=3, mode='reflect') # bi-cubic/reflect is what Matlab imresize() does by default
    # Normalize the two maps to have zero mean and unit std
    map1 = normalize(map1, method='standard')
    map2 = normalize(map2, method='standard')
    # Compute correlation coefficient
    return np.corrcoef(map1.ravel(), map2.ravel())[0,1]


def SIM(saliency_map1, saliency_map2):
    '''
    Similarity between two different saliency maps when viewed as distributions
    (SIM=1 means the distributions are identical).
    This similarity measure is also called **histogram intersection**.
    Parameters
    ----------
    saliency_map1 : real-valued matrix
        If the two maps are different in shape, saliency_map1 will be resized to match saliency_map2.
    saliency_map2 : real-valued matrix
    Returns
    -------
    SIM : float, between [0,1]
    '''
    map1 = np.array(saliency_map1, copy=False)
    map2 = np.array(saliency_map2, copy=False)
    if map1.shape != map2.shape:
        map1 = resize(map1, map2.shape, order=3, mode='reflect') # bi-cubic/reflect is what Matlab imresize() does by default
    # Normalize the two maps to have values between [0,1] and sum up to 1
    map1 = normalize(map1, method='range')
    map2 = normalize(map2, method='range')
    map1 = normalize(map1, method='sum')
    map2 = normalize(map2, method='sum')
    # Compute histogram intersection
    intersection = np.minimum(map1, map2)
    return np.sum(intersection)
