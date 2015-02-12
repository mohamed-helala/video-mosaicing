import cv2.cv as cv
import numpy as np
from camera import *
from dataloader import *
from ps_stereo import *
from pygco import *
from graphcut import *

CAMSFile = "cams.txt"
inputDir = "./data"


def test_load_frames():

    frames = load_frames(inputDir, 3, 0)

    #for k in range(len(frames)):
        #cv.NamedWindow("original" + str(k), 1)
        #cv.ShowImage("original" + str(k), cv.fromarray(frames[k]))
    ##end for

    #cv.WaitKey(0)
    #cv.DestroyAllWindows()
    return frames


def test_load_cams():

    cams = load_cams(inputDir + "/" + CAMSFile)

    #for cam in cams:
        #print cam.__dict__
    #end for
    return cams


def test_adjust_cams():

    cams = test_load_cams()
    v_cam = cams[len(cams) / 2]

    adj_cams = adjust_cams(v_cam, cams)
    #for cam in adj_cams:
        #print cam.E
    return v_cam, adj_cams


def test_gen_plane_homographies():
    v_cam, mod_cams = test_adjust_cams()
    depths = [58.5, 59]  # np.arange(34, 46, 0.5)
    hs_list = gen_plane_homographies(v_cam, mod_cams, depths)
    #print hs_list
    return hs_list, len(depths)


def test_gen_sweep_planes():
    hs_list, nPlanes = test_gen_plane_homographies()
    frames = test_load_frames()
    bldImages = []
    planecosts = []
    maxrecs = []
    for k in range(nPlanes):
        plane_hs = []
        for hs in hs_list:
            plane_hs.append(hs[k])
        #end for
        (bldIm8U, costs, bldrec) = gen_sweep_plane(plane_hs, frames)
        bldImages.append(bldIm8U)
        planecosts.append(costs)
        maxrecs.append(bldrec)
    #end for
    return (bldImages, planecosts, maxrecs)


def test_gen_gco_labels():
    bldImages, planecosts, maxrecs = test_gen_sweep_planes()
    opt_im, labels, max_rec = gen_gco_labels(bldImages, planecosts, maxrecs)

    cv.NamedWindow("original1", 1)
    cv.ShowImage("original1", cv.fromarray(opt_im))
    cv.WaitKey(0)
    cv.DestroyAllWindows()


def test_gco():
    dcosts = np.ones((10, 10, 3), dtype=np.float)
    dcosts = dcosts + np.random.normal(0, 0.5, size=dcosts.shape)
    dcosts = (10 * dcosts).copy("C").astype(np.int32)
    print dcosts
    result = optMosaic(dcosts, 5, -15, 2)
    print result


if __name__ == "__main__":
    #test_load_frames()
    #test_load_cams()
    #test_adjust_cams()
    #test_gen_plane_homographies()
    #test_gen_sweep_planes()
    #test_gco()
    test_gen_gco_labels()