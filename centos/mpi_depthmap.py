import sys
from dataloader import *
from camera import *
from ps_stereo import *
from graphcut import *
from mpi4py import MPI


CAMSFile = "cams.txt"


def gen_mosaic(comm, in_frames, in_cams, in_depths):

    P = comm.Get_size()
    proc_id = comm.Get_rank()
    master = 0

    # Parameters
    loc_frames = None
    hs = []  # homographoies
    h_shape = None  # homographie shape
    h_scat = None  # A list to hold scattered homographies
    nPlanes = None  # Number of depth planes
    b_size = 0  # Processing block size

    if proc_id == master:

        nPlanes = len(in_depths)  # number of depth levels

        # if number of processes > num.of planes then finalize
        proc0_size = (nPlanes) // P
        if(proc0_size == 0):
            print "Too many processes, Number of planes: ", nPlanes, \
                " Number of processes: ", P
            MPI.Finalize()
            comm.exit(1)
        #end if

        ##### Load frames and calculate homographies
        cams = in_cams  # Read Camera File

        loc_frames = in_frames  # Read Frames

        v_cam = cams[len(cams) / 2]
        mod_cams = adjust_cams(v_cam, cams)  # Adjust Cams

        hs_list = gen_plane_homographies(v_cam, mod_cams, in_depths)  # Generate
                                                                # Homographies

        ##### Broadcast frames
        comm.Bcast(np.asarray(len(loc_frames)), master)  # broadcast shape
        comm.Bcast(np.asarray(loc_frames[0].shape), master)  # broadcast shape

        f_lin = np.empty(0, dtype=np.uint8)  # a linear array to hold all frames
        for k in range(len(loc_frames)):
            f_lin = np.concatenate((f_lin, loc_frames[k].ravel()))
        #end if

        comm.Bcast(f_lin, master)  # broadcast frames

        ##### Prepare for scattering plane_homographies
        comm.Bcast(np.asarray(nPlanes), master)  # broadcast hmography shape

        comm.Bcast(np.asarray(hs_list[0][0].shape), master)  # Broadcast
                                                            # homography shape

        block_counts = np.empty(P, dtype=np.int)  # Counts for scatterv
        block_disp = np.empty(P, dtype=np.int)  # Displacements for scatterv
        disp = 0

        block_counts, block_disp = \
                            get_GathScat_info(P, nPlanes, len(loc_frames) *
                                            np.prod((hs_list[0][0].shape)))
        h_lin = get_Scat_data(P, nPlanes, hs_list)
        h_shape = hs_list[0][0].shape
        h_scat = [h_lin, block_counts, block_disp, MPI.DOUBLE]  # scatter info

    else:
        ### Recieve frames
        nframes = np.empty(1, dtype=np.int)
        shape = np.empty(3, dtype=np.int)

        comm.Bcast(nframes, master)  # Recieve frame shape
        comm.Bcast(shape, master)

        fSize = np.prod(shape)  # Size of a single frame
        fsSize = fSize * nframes[0]  # Total size of incomming frames
        f_lin = np.empty(fsSize, dtype=np.uint8)

        comm.Bcast(f_lin, master)  # Recieve frames
        loc_frames = np.split(f_lin, nframes[0])  # Split the linear array

        for k in range(len(loc_frames)):
            loc_frames[k].shape = shape

        ### Recieve homographies
        nPlanesArr = np.empty(1, dtype=np.int)
        comm.Bcast(nPlanesArr, master)
        nPlanes = nPlanesArr[0]

        h_shape = np.empty(2, dtype=np.int)
        comm.Bcast(h_shape, master)  # Recieve homography shape

    #end if

    ##### Scatter plane_homographies
    b_size = BLOCK_SIZE(proc_id, nPlanes, P)
    size = b_size * len(loc_frames) * np.prod((h_shape))
    r_buff = np.empty(size, dtype=np.float)

    comm.Scatterv(h_scat, r_buff, master)  # Do scatterv

    # Split homographies into [nframes, block_size, h_shape[0], h_shape[1]]
    hs = np.split(r_buff, len(loc_frames))

    for k in range(len(loc_frames)):
        hs[k].shape = (b_size, h_shape[0],
                                h_shape[1])
    #end for

    ##### Generate Planes
    local_bldImages = []
    local_plane_costs = []
    local_maxrecs = []
    for k in range(b_size):
        plane_hs = []
        for f_hs in hs:
            plane_hs.append(f_hs[k])
        #end for
        (bldIm8U, costs, bldrec) = gen_sweep_plane(plane_hs, loc_frames)
        local_bldImages.append(bldIm8U)
        local_plane_costs.append(costs)
        local_maxrecs.append(bldrec)
        #print "finishing depth" + str(BLOCK_LO(proc_id, nPlanes, P) + k) + "\n"
    #end for

    ##### Gather maxrecs, plane costs, blended images
    bldImages = []
    plane_costs = []
    maxrecs = []

    comm.Barrier()  # Wait untill all processes finish

    ## Gather maxrecs
    maxrecs_send_buff = np.empty([len(local_maxrecs), 4], dtype=np.float)
    for k in range(len(local_maxrecs)):
        maxrecs_send_buff[k, :] = [local_maxrecs[k].left, local_maxrecs[k].top,
                                local_maxrecs[k].right, local_maxrecs[k].bottom]
    s_gath = None
    maxrecs_rec_Buff = None
    if proc_id == master:
        block_counts, block_disp = get_GathScat_info(P, nPlanes, 4)
        maxrecs_rec_Buff = np.empty([nPlanes * 4], dtype=np.float)
        s_gath = [maxrecs_rec_Buff, block_counts, block_disp, MPI.DOUBLE]
    #end if

    comm.Gatherv(maxrecs_send_buff.ravel(), s_gath, master)

    if proc_id == 0:
        maxrecs_rec_Buff.shape = [nPlanes, 4]
        for k in range(nPlanes):
            maxrecs.append(Rect(Point(maxrecs_rec_Buff[k][0],
                                    maxrecs_rec_Buff[k][1]),
                                Point(maxrecs_rec_Buff[k][2],
                                    maxrecs_rec_Buff[k][3])))
        #end for
    #end if

    ## Gather plane costs
    costs_send_buff = np.empty(0, dtype=np.float)
    for k in range(len(local_plane_costs)):
        costs_send_buff = np.concatenate((costs_send_buff,
                                        local_plane_costs[k].ravel()))
    #end for

    _gath = None
    costs_rec_Buff = None
    if proc_id == 0:
        costs_counts = np.zeros(P, dtype=np.int)
        costs_disp = np.zeros(P, dtype=np.int)
        disp = 0
        b_start = 0
        for k in range(P):
            block_size = BLOCK_SIZE(k, nPlanes, P)
            for w in range(block_size):
                costs_counts[k] += maxrecs[b_start + w].width() * \
                                maxrecs[b_start + w].height()
            #end for
            costs_disp[k] = disp
            disp += costs_counts[k]
            b_start += block_size
        #end for
        costs_rec_Buff = np.empty(np.sum(costs_counts), dtype=np.float)
        _gath = [costs_rec_Buff, costs_counts, costs_disp, MPI.DOUBLE]
    #end if

    comm.Gatherv(costs_send_buff.ravel(), _gath, master)

    if proc_id == 0:
        split_sizes = []
        disp = 0
        for k in range(len(maxrecs) - 1):
            size = maxrecs[k].width() * maxrecs[k].height()
            disp += size
            split_sizes.append(disp)
        #end for

        plane_costs = np.split(costs_rec_Buff, split_sizes)
        for k in range(len(maxrecs)):
            plane_costs[k].shape = [maxrecs[k].width(), maxrecs[k].height()]
            plane_costs[k] = plane_costs[k].astype(int)
        #end for
    #end if

    # Gather blended images

    im_send_buff = np.empty(0, dtype=np.float)
    for k in range(len(local_bldImages)):
        im_send_buff = np.concatenate((im_send_buff,
                                    local_bldImages[k].ravel()))
    #end for
    im_rec_Buff = None
    if proc_id == 0:
        _gath[1] = _gath[1] * 3  # we have 3 channels
        _gath[2] = _gath[2] * 3
        im_rec_Buff = np.empty(np.sum(_gath[1]), dtype=np.float)
        _gath[0] = im_rec_Buff
    #end if

    comm.Gatherv(im_send_buff, _gath, master)

    op_im = None
    depth_map = None
    opt_rec = None
    if proc_id == 0:

        split_sizes = []
        disp = 0
        for k in range(len(maxrecs) - 1):
            size = maxrecs[k].width() * maxrecs[k].height() * 3
            disp += size
            split_sizes.append(disp)
        #end for
        bldImages = np.split(im_rec_Buff, split_sizes)
        for k in range(len(maxrecs)):
            bldImages[k].shape = [maxrecs[k].height(), maxrecs[k].width(), 3]
            bldImages[k] = bldImages[k].astype(np.uint8)
        #end for

        ##### Apply Graphcut
        op_im, depth_map, opt_rec = gen_gco_labels(bldImages,
                                            plane_costs, maxrecs)
    #end if

    # Return the mosaic image, depth mao, and boundary rectangle
    return op_im, depth_map, opt_rec

#end-def-depthmap


def get_GathScat_info(P, n_elem, elem_size):

    block_counts = np.empty(P, dtype=np.int)  # Counts
    block_disp = np.empty(P, dtype=np.int)  # Displacements

    disp = 0
    for k in range(P):
        n_loc_elems = BLOCK_SIZE(k, n_elem, P)
        block_counts[k] = (n_loc_elems * elem_size)
        block_disp[k] = disp
        disp += block_counts[k]
    #end if

    return block_counts, block_disp


# Given a list of arrays of similar size, the function generates
# a linear array by treating the given list as a one array by
# concatenating the colums beside each other on the given order and
# distributing the rows of the all array to the given P processes.
def get_Scat_data(P, n_elems, hs_list):
    h_lin = np.empty(0, dtype=np.int)  # A linear array to scattered data
                                       # According to the counts and disp

    disp = 0
    for k in range(P):
        size = BLOCK_SIZE(k, n_elems, P)
        for k1 in range(len(hs_list)):
            for w in range(size):
                h_lin = np.concatenate((h_lin,
                                        hs_list[k1][w + disp].ravel()))
            #end for
        disp += BLOCK_SIZE(k, n_elems, P)
        #end for
    #end for
    return h_lin
