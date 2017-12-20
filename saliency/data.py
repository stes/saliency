import numpy  as np
import matplotlib.pyplot as plt
import skimage.io
from scipy.io import loadmat
import imageio
import numpy as np
import os
import h5py

VIDEO_FILES = [
    'clip_6.avi',
    'clip_12.avi',
    'clip_17.avi'
]
GAZE_FILES   = [
    'gaze_clip_6.mat',
    'gaze_clip_12.mat',
    'gaze_clip_17.mat',
]

def load_images(root="data/imgs"):

    X = []
    y = []
    src =  ['54.jpg',
            '67.jpg',
            '91.jpg',
            'balloons.png',
            '2sand5s.png',
            'conj2.png',
            'tdo1.png']
    gt = ['d54.jpg', 'd67.jpg', 'd91.jpg', None, None, None, None]

    root = "data/imgs"

    for fname, fname_gt in zip(src, gt):
        fname = os.path.join(root, fname)

        img = skimage.io.imread(fname) / 255.
        X.append(img)

        if fname_gt is not None:
            fname_gt = os.path.join(root, fname_gt)
            img = skimage.io.imread(fname_gt) / 255.
            y.append(img)

        else:
            y.append(None)

    return X, y

def load_video(index, root='data/attention_video_database'):
    """ Load video and ground truth gaze files

    index : Video index, in [0,1,2]
    root  : Root directory for video and ground truth files

    Returns : tuple (X, y) containing video files
        X (frames, width, height, channels)
        y (coordinates, frames, subjects)
    """

    assert os.path.exists(root), "Given root directory does not exist: {}".format(root)
    assert index in list(range(len(VIDEO_FILES))), "Video index out of range"

    fname_video = os.path.join(root, VIDEO_FILES[index])
    fname_gaze  = os.path.join(root, GAZE_FILES[index])

    vid    = imageio.get_reader(fname_video,  'ffmpeg')
    X      = np.stack([vid.get_data(num) for num in range(vid.get_length())], axis=0)
    y      = loadmat(fname_gaze)['gaze']

    return X, y
