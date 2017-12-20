import skimage.color
from scipy.ndimage.filters import convolve as conv2d

import numpy  as np
import skimage.io

import skimage.transform
import skimage.filters
from scipy.stats import norm

import utils

### Helper functions ###

def construct_gabor(angle, phase=0., gamma=1, stddev=10**.5, wavelen=3.5):
    """
    angle
    phase
    gamma   : aspect ratio
    stddev  : standart deviation
    wavelen : wavelength
    """

    n_stddev = 4

    sz = np.ceil(n_stddev*stddev).astype(int)

    xx, yy = [x.flatten() for x in np.meshgrid(np.linspace(-sz,sz,2*sz+1), np.linspace(-sz,sz,2*sz+1))]

    xx_ =   xx * np.cos(angle) + yy * np.sin(angle)
    yy_ = - xx * np.sin(angle) + yy * np.cos(angle)

    expfilter = np.exp( (xx_**2 + gamma**2 * yy_**2)  /  (-2*stddev**2))
    sinfilter = np.cos(2*np.pi / wavelen * xx_ + phase)

    return (expfilter * sinfilter).reshape(2*sz+1,2*sz+1)

def attenuate_borders(img, borderSize):
    w, h, _ = img.shape

    if (borderSize * 2 > w):
        borderSize = np.floor(h / 2)
    if (borderSize * 2 > h):
        borderSize = np.floor(w / 2)
    if (borderSize < 1):
        return

    dampening = np.linspace(0,1,borderSize)

    img[:borderSize,:]  *= dampening[:,np.newaxis,np.newaxis]
    img[-borderSize:,:] *= dampening[::-1,np.newaxis,np.newaxis]
    img[:,:borderSize]  *= dampening[np.newaxis,:,np.newaxis]
    img[:,-borderSize:] *= dampening[np.newaxis,::-1,np.newaxis]

    return img

def center_bias(maps, length = 3):

    w,h = maps.shape[:2]

    xx = norm.pdf(np.arange(w) - w/2, 0, w/length)
    yy = norm.pdf(np.arange(h) - h/2, 0, h/length)

    bias = xx[:,np.newaxis] * yy[np.newaxis,:]
    bias /= bias.sum()

    return bias * maps

### (Non-) linear filter functions ###

def intensity(img):
    """ Compute Intensity Features"""
    return img.mean(axis=-1, keepdims=True)

def gabor(img, n_gaborangles=4, phases=[0, np.pi/2]):
    """ Filter image with gabor filter array """

    # construct filters
    max_angle = np.pi - (np.pi / n_gaborangles)
    angles = np.linspace(0,max_angle,n_gaborangles)

    filters = np.stack([construct_gabor(angle, phase)
                            for angle in angles
                            for phase in phases], axis=-1)

    # given n_gaborangles filters and img.shape[2] image channels, compute
    # all combinations

    return  np.stack([conv2d(img[...,c], filters[...,k], mode='reflect')
                        for c in range(img.shape[2])
                        for k in range(filters.shape[2])], axis=-1)

def rgby(img, threshold=.1):
    """ Compute Color Opponent Features """
    r,g,b  = list(img.transpose((2,0,1)))
    maxrgb = np.max(img, axis=-1)
    mask   = (maxrgb > threshold)+0.
    maxrgb[mask==0] = 1.

    RG = mask * (r - g) / maxrgb
    BY = mask * (b - np.min(img[...,:2], axis=-1)) / maxrgb

    return np.stack([RG, BY], axis=-1)

def colorbias(img, refcolor=np.array([1.,0,0])):
    """ Compute Color Bias """
    img_hsv  = skimage.color.rgb2hsv(img)
    refcolor = skimage.color.rgb2hsv(refcolor.reshape(1,1,3)) # to make it compatible
    #dH = np.abs(np.sin((img_hsv[...,0] - refcolor[...,0])))
    #dS = np.abs(img_hsv[...,1] - refcolor[...,1])
    #dV = np.abs(img_hsv[...,2] - refcolor[...,2])

    hsv2xyz = lambda h,s,v : np.stack([s*np.sin(h*2*np.pi), s*np.cos(h*2*np.pi), v], axis=-1)

    xyz_ref = hsv2xyz(*refcolor.transpose((2,0,1)))
    xyz_img = hsv2xyz(*img_hsv.transpose((2,0,1)))

    return 1 - ((xyz_ref - xyz_img)**2).sum(axis=-1, keepdims=True)**.5

def collect_maps(img, fns=[intensity,gabor,rgby,colorbias], *args, **kwargs):
    return np.concatenate([fn(img) for fn in fns], axis=-1)

def resize(img, width=64):
    imsize = [ width, round(img.shape[1] / img.shape[0] * width) ]
    return skimage.transform.resize(img, imsize,mode='reflect',preserve_range=True)

def peakiness(img):
    img_norm = (img  - img.min()) / (img.max() - img.min())
    locmax_avg, locmax_num, _ = local_maxima(img_norm, 0.1);

    if locmax_num > 1:
        p = (1 - locmax_avg)**2
    else:
        p = 1

    return p

def local_maxima(data, threshold):

    refData = data[1:-1,1:-1]
    ii,jj = np.where(  (refData >= data[:-2,1:-1]) & \
                       (refData >= data[2:,1:-1])  & \
                       (refData >= data[1:-1,:-2]) & \
                       (refData >= data[1:-1,2:])  & \
                       (refData >= threshold))

    maxData = refData[ii,jj]

    lm_avg = np.mean(maxData);
    lm_sum = np.sum(maxData);
    lm_num = len(maxData);

    return lm_avg, lm_num, lm_sum

def saliency(img, surround_sig = [ 2, 8], subtract_min=True, norm_weights=True):
    if subtract_min:
        norm = lambda x : (x  - x.min()) / (x.max() - x.min())
    else:
        norm = lambda x : x / max(img)

    summap = 0;
    for ssig in surround_sig:
        map_ = norm(skimage.filters.gaussian(img, sigma=ssig, truncate=2, mode="reflect"))
        wt_ = peakiness(map_) if norm_weights else 1
        summap = summap + map_ * wt_

    return norm(summap)

### Sequential saliency

def sequential_salicency(salmap, length, inhibition_strategy='gaussian', sampling_strategy='max'):
    """ Implementation of inhibition by return
    """

    current_map = salmap.copy()
    inhibition  = np.zeros_like(salmap)
    fixation = np.zeros((length, 2))

    for t in range(length):
        x,y = utils.argmax2d(current_map)
        fixation[t,:] = np.array([x,y])
        inhibition[:,:] = 0
        inhibition[x,y] = 1
        inhibition = skimage.filters.gaussian(inhibition, sigma=2, mode='reflect')
        inhibition = 1 - inhibition / inhibition.max()
        current_map *= inhibition

    return fixation, current_map
