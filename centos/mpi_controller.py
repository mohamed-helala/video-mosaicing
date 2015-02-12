import sys
import numpy as np
from mpi4py import MPI
from util import *
from camera import *
from mpi_depthmap import *
from dataloader import *


CAMSFile = "cams.txt"


def mosaic_controller():

    comm = MPI.COMM_WORLD
    P = comm.Get_size()
    proc_id = comm.Get_rank()
    master = 0  # The rank of the global master process

    inputDir = sys.argv[1]  # Input directory
    n_comms = np.empty([1], dtype=int)
    depth_info = None  # Input depths
    cluster_size = None  # Input cluster size
    n_clusters = None  # Total number of clusters
    n_lo_clusters = None  # Number of clusters in an assigned block

    lo_camClusters = None  # List of clusters processed by a certain sub. comm.
    lo_frameClusters = None  # List of frames for each cluster.

    if proc_id == master:
        try:
            P_cluster = int(sys.argv[2])  # number pf processes
                                          # to serve each cluster
            cluster_size = int(sys.argv[3])  # Size of each cluster
            d_start = float(sys.argv[4])  # start of the depth range
            d_end = float(sys.argv[5])  # end of the depth range
            d_step = float(sys.argv[6])  # the depth step
            # depths of sweep planes
            depth_info = np.arange(d_start, d_end, d_step)
        except:
            print "Usage:", sys.argv[0], "input_directory ", \
                    "n_processes_per_cluster"
            MPI.Finalize()
            sys.exit(1)

        # Define the number of parallel PSS instances
        n_comms[0] = P // P_cluster
    #end if

    comm.Barrier()
    elapsed_time = np.zeros([1], dtype=float)
    elapsed_time[0] = - MPI.Wtime()
    comm.Bcast(n_comms, master)

    # define the owner of the current process
    owner = BLOCK_OWN(proc_id, P, n_comms[0])

    # Split the current communicator into different subgroubs specidied by
    # the owner
    comm2 = comm.Split(owner, proc_id)

    # Get the proc_id and the size of the current process communicator
    cproc_id = comm2.Get_rank()
    cP = comm2.Get_size()

    # Get the index of the local master process of the current communicator
    c_master = BLOCK_LO(owner, P, n_comms[0])

    # Now load the input frames, cluster them and distribute the clusters
    # to the set of communicators
    if proc_id == master:

        cams = load_cams(inputDir + "/" + CAMSFile)  # Read Camera File

        frames = load_frames(inputDir, len(cams), 0)  # Read Frames

        # send the depth info to the local master processes of each PSS instance
        for k in range(n_comms[0]):
            m = BLOCK_LO(k, P, n_comms[0])
            if(m != master):
                comm.Send(np.asarray([len(depth_info)], dtype=int), dest=m)
                comm.Send(depth_info, dest=m)
            #end if
        #end for

        # Generate clusters using an overlapped sliding window
        n = len(cams)
        c = (cluster_size - 1) // 2
        frame_clusters = []
        cams_clusters = []
        for k in range(n):
            if k % c == 0 and k > 0:
                #print str(k - c), str(k + c + 1), "\n"
                frame_clusters.append(frames[k - c:k + c + 1])
                cams_clusters.append(cams[k - c:k + c + 1])
            #end if
        #end for
        n_clusters = len(cams_clusters)

        # Distrbute the clusters to the local masters of all PSS instances
        disp = 0
        for k in range(n_comms[0]):
            m = BLOCK_LO(k, P, n_comms[0])
            b_size = BLOCK_SIZE(k, n_clusters, n_comms[0])
            m_camClusters = cams_clusters[disp:disp + b_size]
            m_frameClusters = frame_clusters[disp:disp + b_size]

            if(m != master):
                #send to masters
                cam_lin = np.empty([0], dtype=np.float)
                frames_lin = np.empty([0], dtype=np.int)
                C_sizes = np.empty([b_size], dtype=np.int)
                for l in range(b_size):
                    c_cam_lin = np.empty([0], dtype=np.float)
                    c_frames_lin = np.empty([0], dtype=np.int)
                    for w in range(len(m_camClusters[l])):
                        c_cam_lin = np.concatenate((c_cam_lin,
                                            m_camClusters[l][w].linearize()))
                        c_frames_lin = np.concatenate((c_frames_lin,
                                            m_frameClusters[l][w].ravel()))
                    #end for
                    C_sizes[l] = len(m_camClusters[l])
                    cam_lin = np.concatenate((cam_lin, c_cam_lin))
                    frames_lin = np.concatenate((frames_lin, c_frames_lin))
                #end for

                comm.Send(np.asarray(b_size, dtype=np.int), dest=m)
                comm.Send(C_sizes, dest=m)

                comm.Send(np.asarray(m_camClusters[0][0].linear_size(),
                                    dtype=np.int), dest=m)
                comm.Send(np.asarray(m_frameClusters[0][0].shape,
                                    dtype=np.int), dest=m)
                comm.Send(cam_lin, dest=m)
                #print cam_lin.shape
                comm.Send(frames_lin, dest=m)

            else:
                lo_camClusters = m_camClusters
                lo_frameClusters = m_frameClusters
                n_lo_clusters = len(m_frameClusters)
            #end if
            disp += b_size
        #end for
    #end if

    # If this process is a local master process then
    # recieve the data from the global master process
    if (proc_id == c_master) and (proc_id != master):
        nPlanes = np.empty([1], dtype=int)
        comm.Recv(nPlanes, source=master)
        depth_info = np.empty([nPlanes[0]], dtype=float)
        comm.Recv(depth_info, source=master)

        n_lo_clusters = np.empty([1], dtype=np.int)
        frame_Shape = np.empty([3], dtype=np.int)
        cam_lin_size = np.empty([1], dtype=np.int)

        comm.Recv(n_lo_clusters, source=master)
        n_lo_clusters = n_lo_clusters[0]

        C_sizes = np.empty([n_lo_clusters], dtype=np.int)

        comm.Recv(C_sizes, source=master)

        comm.Recv(cam_lin_size, source=master)
        comm.Recv(frame_Shape, source=master)

        cC_lin = np.empty([cam_lin_size[0] * np.sum(C_sizes)], dtype=np.float)
        fC_lin = np.empty([np.prod(frame_Shape) * np.sum(C_sizes)],
                             dtype=np.int)

        comm.Recv(cC_lin, source=master)
        comm.Recv(fC_lin, source=master)

        ## Form the local clusters
        cC_idxs = []
        fC_idxs = []
        disp_cC = 0
        disp_fC = 0
        for k in range(n_lo_clusters - 1):
            disp_cC += C_sizes[k] * cam_lin_size[0]
            disp_fC += C_sizes[k] * np.prod(frame_Shape)
            cC_idxs.append(disp_cC)
            fC_idxs.append(disp_fC)
        #end for
        lin_cams = np.split(cC_lin, cC_idxs)
        lin_frames = np.split(fC_lin, fC_idxs)

        lo_camClusters = []
        lo_frameClusters = []
        for k in range(n_lo_clusters):
            frames = np.split(lin_frames[k], C_sizes[k])
            cams = np.split(lin_cams[k], C_sizes[k])
            cam_info = []
            for w in range(C_sizes[k]):

                frames[w] = frames[w].astype(np.uint8)
                frames[w].shape = frame_Shape
                camIn = CamInfo().create_from_linear(cams[w],
                            ((3, 3), (3, 4), (3, 4), (3, 1), (3, 1)))
                cam_info.append(camIn)
            #end for
            lo_camClusters.append(cam_info)
            lo_frameClusters.append(frames)

        #end for
    #end if

    # We need to brodcast the local master proc_id of each communicator to
    # all processes within the communicator
    b_master = np.zeros([1], dtype=int)
    b_master[0] = c_master if proc_id == c_master else -1
    n_lo_clustersA = np.zeros([1], dtype=int)
    n_lo_clustersA[0] = n_lo_clusters if proc_id == c_master else 0

    comm2.Barrier()
    comm2.Bcast(b_master, 0)
    comm2.Bcast(n_lo_clustersA, 0)
    n_lo_clusters = n_lo_clustersA[0]
    del n_lo_clustersA

    # Now we can compare the locsl master proc_id in b_master to the one
    # calculated in c_master to make sure that only the processes of each
    # communicator will succeed in the following condition.
    op_images = []
    op_recs = []
    if c_master == b_master[0]:
        # Loop on the assigned clusters
        for k in range(n_lo_clusters):
            comm2.Barrier()
            cFrames = lo_frameClusters[k] if proc_id == c_master else None
            cCams = lo_camClusters[k] if proc_id == c_master else None

            ###########  Apply the parallel PSS algorithm
            op_im, _, opt_rec = gen_mosaic(comm2,
                                        cFrames, cCams, depth_info)

            if(proc_id == c_master):
                # Collect the mosaic image of each cluster.
                op_images.append(op_im)
                op_recs.append(opt_rec)

            #end if
        #end for
    #end if

    if proc_id == c_master and proc_id != master:
        # Send the accumulated mosaics and their shapes to the global
        # master process.
        recs_lin = np.empty([0], dtype=np.float)
        opIm_lin = np.empty([0], dtype=np.uint8)

        for k in range(len(op_recs)):
            recs_lin = np.concatenate((recs_lin, np.asarray([op_recs[k].left,
                        op_recs[k].top, op_recs[k].right, op_recs[k].bottom])))
            opIm_lin = np.concatenate((opIm_lin, op_images[k].ravel()))
        #end for
        comm.Send(recs_lin, dest=master)
        comm.Send(opIm_lin, dest=master)

    else:
        if proc_id == master:  # Enter the global master process
            # Collect the accumulated mosaics and their shapes
            acc_mosaics = op_images
            acc_recs = op_recs
            for k in range(n_comms[0]):
                m = BLOCK_LO(k, P, n_comms[0])
                if(m != master):
                    b_size = BLOCK_SIZE(k, n_clusters, n_comms[0])
                    recs_lin = np.empty([b_size * 4], dtype=np.float)
                    comm.Recv(recs_lin, source=m)
                    recs_info = np.split(recs_lin, b_size)
                    mShapes = []
                    f_lin_size = 0
                    for w in range(len(recs_info)):
                        rec = Rect(
                                    Point(recs_info[w][0], recs_info[w][1]),
                                    Point(recs_info[w][2], recs_info[w][3]))
                        shape = [rec.height(), rec.width(), 3]
                        mShapes.append(shape)
                        f_lin_size += np.prod(shape)
                        acc_recs.append(rec)
                    #end for
                    mosaics_lin = np.empty([f_lin_size], dtype=np.uint8)
                    comm.Recv(mosaics_lin, source=m)

                    mC_idxs = []
                    disp_fC = 0
                    for k in range(b_size - 1):
                        disp_fC += np.prod(mShapes[k])
                        mC_idxs.append(disp_fC)
                    #end for
                    mosaics = np.split(mosaics_lin, mC_idxs)
                    for k in range(b_size):
                        mosaics[k].shape = mShapes[k]
                        acc_mosaics.append(mosaics[k])
                    #end for
                #end if
            #end for
            #################### Save the mosaics to the output folder
            save_images(acc_mosaics, "./mosaics")
        #end if
    #end if

    comm.Barrier()
    elapsed_time[0] += MPI.Wtime()
    total_time = np.zeros([1], dtype=float)
    comm.Reduce(elapsed_time, total_time, op=MPI.SUM, root=master)

    if proc_id == master:
        print "time: " + str(total_time[0] / P)


if __name__ == "__main__":
    mosaic_controller()
