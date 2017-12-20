import numpy  as np

def argmax2d(X):

    if X.ndim == 2:
        xx, yy = argmax2d(X[np.newaxis,:,:])
        return xx[0], yy[0]

    # https://stackoverflow.com/questions/30589211/numpy-argmax-over-multiple-axes-without-loop
    # Reshape input array to a 2D array with rows being kept as with original array.
    # Then, get idnices of max values along the columns.
    max_idx = X.reshape(X.shape[0],-1).argmax(1)

    # Get unravel indices corresponding to original shape of A
    maxpos_vect = np.column_stack(np.unravel_index(max_idx, X[0,:,:].shape))
    xx, yy = maxpos_vect.T

    return xx, yy

def mse(salmap, groundtruth, per_frame=True):
    """ Compute Mean Squared Error Score """

    median = np.nanmedian(groundtruth, axis=-1)
    xx, yy = utils.argmax2d(salmap)

    mse_score = ((xx - median[0]) ** 2 + (xx - median[1])**2)

    if per_frame:
        return np.nanmean(mse_score), mse_score
    return np.nanmean(mse_score)

def nss(S_, y, normalized=False, per_frame=True):
    """ Compute NSS Score """
    if not normalized:
        salmap = (salmap - salmap.mean(axis=(1,2), keepdims=True)) / salmap.std(axis=(1,2), keepdims=True)

    y_ = y.copy()
    y_[0] = np.clip(y_[0], 0, S_.shape[1]-1)
    y_[1] = np.clip(y_[1], 0, S_.shape[2]-1)

    iframe,isub = np.where(~np.isnan(y.sum(axis=0)))

    nss_score = np.zeros(y[0].shape)
    nss_score[:] = np.nan
    nss_score[iframe, isub] = S_[iframe, y_[0,iframe,isub].astype("int"), y_[1,iframe,isub].astype("int")]
    nss_score = np.nanmean(nss_score, axis=1)

    if per_frame:
        return np.nanmean(nss_score), nss_score
    return np.nanmean(nss_score)
