import cv2.cv as cv
import numpy as np
from camera import *


def load_cams(cams_file):  # load all cameras

        # open file for reading
        camsfile = open(cams_file, 'r')
        cams = []
        idx = -1
        for line in camsfile:  # browse the file line by line
            if line.startswith('#'):
                _, idxval = line.split()
                idx = int(idxval)
                cams.append(CamInfo())
                continue
            #end if

            cinfo = cams[idx]
            elems = line.split()

            if line.startswith('K'):
                data = np.array(elems[1:], float)
                data.shape = (3, 3)
                cinfo.K = data
            elif line.startswith('E'):
                data = np.array(elems[1:], float)
                data.shape = (3, 4)
                cinfo.E = data
            elif line.startswith('P'):
                data = np.array(elems[1:], float)
                data.shape = (3, 4)
                cinfo.P = data
            elif line.startswith('r'):
                data = np.array(elems[1:], float)
                data.shape = (3, 1)
                cinfo.r = data
            elif line.startswith('t'):
                data = np.array(elems[1:], float)
                data.shape = (3, 1)
                cinfo.t = data
            #end if

        #end for

        return cams


def save_images(images, out_dir):

    for k in range(len(images)):
        cv.SaveImage(out_dir + "/" + str(k) + ".jpg", cv.fromarray(images[k]))
    #end for


def load_frames(inputDir, nframes, level):
    # Load images from the input directory
    frames = []
    for k in range(nframes):
        im = cv.LoadImageM(inputDir + "/" + str(level) + str(k) + ".jpg",
                              cv.CV_LOAD_IMAGE_COLOR)
        frames.append(np.asarray(im))
    #end for

    return frames