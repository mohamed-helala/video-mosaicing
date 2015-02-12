import sys
from dataloader import *
from camera import *
from ps_stereo import *
from graphcut import *


CAMSFile = "cams.txt"


def gen_mosaic(frames, cams, depths):

    # Adjust Cams
    v_cam = cams[len(cams) / 2]
    mod_cams = adjust_cams(v_cam, cams)

    # Generate Homographies
    hs_list = gen_plane_homographies(v_cam, mod_cams, depths)

    # Generate Planes
    bldImages = []
    planecosts = []
    maxrecs = []
    for k in range(len(depths)):
        plane_hs = []
        for hs in hs_list:
            plane_hs.append(hs[k])
        #end for
        (bldIm8U, costs, bldrec) = gen_sweep_plane(plane_hs, frames)
        bldImages.append(bldIm8U)
        planecosts.append(costs)
        maxrecs.append(bldrec)
    #end for

    # Apply Graphcut
    op_im, depth_map, opt_rec = gen_gco_labels(bldImages, planecosts, maxrecs)

    # Return the depth image and its depth map
    return op_im, depth_map, opt_rec

if __name__ == "__main__":
    depthmap()
