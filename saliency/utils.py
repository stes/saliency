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
