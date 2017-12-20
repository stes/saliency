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

def load_fixations(model, root):
    """ Load pre-computed fixations for the frames in the Video
    """
    pass




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
