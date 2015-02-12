import numpy as np
import numpy.linalg as linalg
import cv2.cv as cv
import cv2
from camera import *
from Polygon import *
from util import Point, Rect


# generate the plane homographies parallel to the given v_cam and
# returns a 3D array [len(src_cams),homog_dim1, homog_dim2]
# representing the plane homographies for each source camera
def gen_plane_homographies(v_cam, src_cams, depths):
    nframes = len(src_cams)
    nplanes = len(depths)
    H_cams = []
    nt = np.array([0, 0, -1], float)
    nt.shape = (1, 3)
    for k in range(nframes):
        H = np.zeros([nplanes, 3, 3], dtype=float)
        R = src_cams[k].E[:3, :3]
        t = src_cams[k].E[:3, 3]
        t.shape = (3, 1)

        for p in range(nplanes):
            d = depths[p]
            if d == 0:
                d = 0.01
            #end if
            M1 = np.dot(t, nt) / d
            M2 = np.dot(R - M1, linalg.inv(src_cams[k].K))
            M2 = np.dot(src_cams[k].K, M2)
            H[p, :3, :3] = linalg.inv(M2)[:3, :3]
        #end for
        H_cams.append(H)
    #end for
    return H_cams


# generate the blended image and color costs for a certain plane,
# given the plane homographies and source frames. Note that
# len(src_Hs) = len(src_frames) or a homography for each frame.
def gen_sweep_plane(src_Hs, src_frames):

    max_poly = None  # the bound of the blended image
    bldIm = None  # the blended plane image
    cost_info = None  # An array structure to collect pixel colors of
                      # blended frames
    nframes = len(src_frames)
    o = np.array([[0], [0], [1]])  # origin = zero

    # Get max polygon
    for k in range(nframes):
        cpoly = Rect(Point(0, 0),
                       Point(src_frames[k].shape[1],
                       src_frames[k].shape[0])).get_trans_regpoly(src_Hs[k], 10)
        if(k == 0):
            max_poly = cpoly
        else:
            max_poly = max_poly | cpoly
        #end if
    #end for

    (xmin, xmax, ymin, ymax) = max_poly.boundingBox(0)
    max_rec = Rect(Point(xmin, ymin), Point(xmax, ymax))
    #print "Max: " , max_rec, "W: ", max_rec.width(), " H: ", max_rec.height()

    #alocate costs and count matrixes
    cost_info = np.zeros([int(max_rec.height()),
                        int(max_rec.width()),
                        nframes, 3], dtype=np.int)
    counts = np.zeros([int(max_rec.height()),
                        int(max_rec.width())], dtype=np.int)

    bldIm = cv.CreateMat(int(np.round(max_rec.height())),
                        int(np.round(max_rec.width())), cv.CV_32FC3)
    cv.Zero(bldIm)

    for k in range(nframes):
        cur_H = src_Hs[k]
        cur_o = np.dot(cur_H, o)
        #  translate the warped frame to origin from topleft corner
        disp = np.array([[1, 0, (0 - cur_o[0, 0]) / cur_o[2, 0]],
                        [0, 1, (0 - cur_o[1, 0]) / cur_o[2, 0]],
                        [0, 0, 1]])
        o_H = np.dot(disp, cur_H)
        tpoly = Rect(Point(0, 0),
                       Point(src_frames[k].shape[1],
                             src_frames[k].shape[0])).get_trans_poly(cur_H)
        cpoly = Rect(Point(0, 0),
                       Point(src_frames[k].shape[1],
                       src_frames[k].shape[0])).get_trans_regpoly(cur_H, 10)
        (xmin, xmax, ymin, ymax) = cpoly.boundingBox(0)
        frec = Rect(Point(xmin, ymin), Point(xmax, ymax))

        mask = gen_mask(frec, tpoly)

        #print "Rec: ", frec, "W: ", frec.width(), " H: ", frec.height()

        if k == 0:
            fwarp = cv2.warpPerspective(src_frames[k], o_H,
                            (int(frec.width()), int(frec.height())))
            # get blended image
            blend_views(bldIm, fwarp, mask,
                            frec, max_rec)
            # determine costs
            collect_costs_info(cost_info, counts, fwarp, mask,
                                frec, max_rec, k)
        else:
            fwarp = cv2.warpPerspective(src_frames[k], o_H,
                            (int(frec.width()), int(frec.height())))

            # get blended image
            blend_views(bldIm, fwarp, mask,
                            frec, max_rec)
            # determine costs
            collect_costs_info(cost_info, counts, fwarp, mask,
                                frec, max_rec, k)
        #end if
    #end for

    bldIm = np.asarray(bldIm)

    # Scale blended image to 8U
    bldIm8U = scaleTo8U(bldIm, counts)

    ## Measure Cost
    costs = None
    costs = calculate_costs(cost_info, counts)

    ##return blended image and costs
    return (bldIm8U, costs, max_rec)


def blend_views(bldIm, frame, mask, frec, max_rec):
    maskMat = cv.fromarray(mask)

    dispX = int(np.round(frec.left - max_rec.left))
    dispY = int(np.round(frec.top - max_rec.top))

    bldROI = cv.GetImage(bldIm)

    cv.SetImageROI(bldROI, (dispX, dispY,
                            int(np.round(frec.width())),
                            int(np.round(frec.height()))))
    cv.Add(bldROI, cv.fromarray(frame), bldROI, maskMat)
    cv.ResetImageROI(bldROI)
    cv.Copy(bldROI, bldIm)

    return bldIm


def collect_costs_info(cost_info, counts, frame, mask, frec, max_rec, idx):

    # Now convert incoming frame to HSV color space and
    # copy the color of each pixel to the corresponding
    # location in newInfo
    hsvIm = cv.fromarray(frame)
    cv.CvtColor(hsvIm, hsvIm, cv.CV_RGB2HSV)
    frame_hsv = np.asarray(hsvIm)

    dispX = int(np.round(frec.left - max_rec.left))
    dispY = int(np.round(frec.top - max_rec.top))

    blndMask = np.zeros(counts.shape, dtype=int)

    cost_info[dispY:int(dispY + np.round(frec.height())):,
                dispX:int(dispX + np.round(frec.width())):, idx, :3] \
            = frame_hsv[:, :, :3]

    blndMask[dispY:int(dispY + np.round(frec.height())):,
                dispX:int(dispX + np.round(frec.width())):] \
            = mask[:, :]
    blndMask = blndMask > 0

    counts[blndMask] = counts[blndMask] + 1

    return cost_info


def calculate_costs(cost_info, counts):
    c = np.std(cost_info, axis=2)
    s = np.sum(c, axis=2)
    cc = counts.copy()
    cc[cc == 0] = 1
    costs = np.where(cc < 3, 500, np.round(s / cc))
    del cc
    costs = np.transpose(costs).astype(int)
    return costs


def scaleTo8U(bldIm, counts):
    w = bldIm.shape[1]
    h = bldIm.shape[0]

    mat = np.zeros((h, w, bldIm.shape[2]), dtype=np.int32)
    mat[:, :, 0] = np.where(counts == 0, 1, counts)
    mat[:, :, 1] = np.where(counts == 0, 1, counts)
    mat[:, :, 2] = np.where(counts == 0, 1, counts)

    mat = cv.fromarray(mat)
    cvbldIm = cv.fromarray(bldIm)
    bldIm8U = cv.CreateMat(h, w, cv.CV_8UC3)
    cv.Div(cvbldIm, mat, cvbldIm)
    cv.Scale(cvbldIm, bldIm8U, 1.0, 0)

    return np.asarray(bldIm8U)


def gen_mask(frec, tpoly):

    mask = cv.CreateMat(int(np.round(frec.height())),
                        int(np.round(frec.width())), cv.CV_8UC1)
    cv.Zero(mask)
    pts = tuple((int(np.round(x - frec.left)), int(np.round(y - frec.top)))
                            for (x, y) in tpoly.contour(0))
    cv.FillPoly(mask, [pts], cv.Scalar(1), lineType=8, shift=0)

    return np.asarray(mask)
