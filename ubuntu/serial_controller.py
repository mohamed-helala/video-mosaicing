import sys
from util import *
from camera import *
from depthmap import *
from dataloader import *
from time import time

CAMSFile = "cams.txt"


def serial_controller():

    try:
        inputDir = sys.argv[1]  # input directory
        cluster_size = int(sys.argv[2])  # Size of each cluster
        d_start = float(sys.argv[3])  # start of the depth range
        d_end = float(sys.argv[4])  # end of the depth range
        d_step = float(sys.argv[5])  # the depth step
        depth_info = np.arange(d_start, d_end, d_step)  # depths of sweep planes
    except:
        print "Usage:", sys.argv[0], "input_directory ", \
                "n_processes_per_cluster"
        sys.exit(1)

    t0 = - time()

    # Load input frames and their associated cameras
    cams = load_cams(inputDir + "/" + CAMSFile)  # Read Camera File

    frames = load_frames(inputDir, len(cams), 0)  # Read Frames

    # Generate clusters using an overlapped sliding window
    c = (cluster_size - 1) // 2
    frame_clusters = []
    cams_clusters = []
    for k in range(len(cams)):
        if k % c == 0 and k > 0:
            #print str(k - c), str(k + c + 1), "\n"
            frame_clusters.append(frames[k - c:k + c + 1])
            cams_clusters.append(cams[k - c:k + c + 1])
        #end if
    #end for
    n_clusters = len(cams_clusters)

    # Now we can compare the master proc_id in b_master to the one calculated
    # in c_master to make sure that only the processes of each communicator will
    # succeed in the following condition.
    op_images = []
    op_recs = []

    # Loop on all clusters
    for k in range(n_clusters):
        cFrames = frame_clusters[k]
        cCams = cams_clusters[k]

         ###########  Apply the serial PSS algorithm
        op_im, _, opt_rec = gen_mosaic(cFrames, cCams, depth_info)

        # Collect the mosaic image of each cluster.
        op_images.append(op_im)
        op_recs.append(opt_rec)
    #end for

    # Save the generated mosaics to the output folder
    save_images(op_images, "./mosaics")

    t0 += time()
    print "time: " + str(t0)


if __name__ == "__main__":
    serial_controller()
