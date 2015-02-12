Welcome to the Video Mosaic Project!
===================================

The video mosaic project extends the technique of (1) to apply plane sweep
stereo on a set of clusters of a long camera motion trajectory.

Contents:
    
    [1] Project Modules
    
    [2] 3rd Party Libraries
    
    [3] Using the Makefile
    
    [4] References
    
    [5] Contact


[1] Project Modules
--------------------------

[1.1] dataloader.py
-----------------
This module supports the loading of project data from disk. It contains two
main functions:
    
    [a] load_cams(cams_file): Loads the input motion trajectory, consisting of
        the camera parameters for each input frame.

    [b] load_frames(inputDir, nframes, level): loads the input frames
        (0 to (nframes - 1)) from the input directory (inputDir) and in the same
        level. Note that because we have multiple mosaic levels, we name the
        frame files as [level + ""  + frame_number].jpg
    [c] save_images(images, out_dir): Saved each image in the given list of images
        to the output directory.

[1.2] camera.py
-------------
This module contains the camera handling functions:
    
    [a] CamInfo class : Contain the camera matrix parameters
        K: intrinsics, r: rotation vector, t: translation vector, E: extrinsic
        P: projection.

    [b] adjust_cams(v_cam, src_cams): adjusts the given source camera by
        transforming the set of cameras in src_cams to the coordinate of camera
        v_cam and returns the new adjusted cameras.
        
    [c] linear_size(): returns the linear size of all camera attributes
        (K, P,E, r, t)
        
    [d] linearize(): returns a linearized array of the camera attributes
    
    [e] create_from_linear(lin_arr, shapes): copy the data from the given
        linear array to the camera matrices in the order (K, P,E, r, t) and
        according to the given shapes.

[1.3] graphcut.py
---------------
This module contains functions related to graph-cut optimization:
    
    [a] gen_gco_labels(bldImages, costs, maxrecs): This function takes a list
        of grid arrays, one for each label. Each array represents the costs
        of its assigned label. The given list has the structure
        [nLabels, w, h]. It also takes a list of rectangle containing a
        rectangle for each grid array representing its 2D position
        with respect to other grids (This is too important for aligning)

[1.4] ps_stereo.py
---------------
This module implements the plane sweep stereo functions:
    
    [a] gen_plane_homographies(v_cam, src_cams, depths): generate the plane
        homographies parallel to the given v_cam and returns a 3D array
        [len(src_cams),homog_dim1, homog_dim2] representing the plane
        homographies for each source camera.

    [b] gen_sweep_plane(src_Hs, src_frames): generate the blended image and
        color costs for a certain plane, given the plane homographies and
        source frames. Note that len(src_Hs) = len(src_frames) or a
        homography for each frame.

    [c] blend_views(bldIm, frame, mask, frec, max_rec): accumulates the blend
        image (bldIm) by adding the warped frame (frame) according to the mask
        (mask). Note that the frame rectangle (frec) and max rectangle (max_rec)
        parameters identify the location of the new frame in the blend image.

    [d] collect_costs_info(cost_info, counts, frame, mask, frec, max_rec, idx):
        accumulates the cost information for the blended image. cost_info is
        an array structure with the shape [max_rec.height(), max_rec.width(),
        nframes, 3] that holds the colors (3 channels) of the warped frames.
        counts identify the number of added colors in each [x, y] location
        of cost_info. The parameters (frame, mask, frec, max_rec) are similar to
        the previous function. idx identify the added frame index in cost_info.

    [e] calculate_costs(cost_info, counts): calculate the variance of each pixel
        in the blended image and return a grid of costs of the same size as the
        blended image. The parameters (cost_info, counts) are similar to
        the previous function.

    [f] scaleTo8U(bldIm, counts): Scales the blended image to 8U according to
        the color count of each pixel.

    [g] gen_mask(frec, tpoly): returns a mask representing the pixels of the
        true boundary (tpoly) in the rectangular boundary (frec).

[1.5] serial_controller.py
---------------
The starting module for the serial program. It include only one starting function:
        
        [a] serial_controller(): This function loads frames and their associated
            cameras from disk and cluster them into a set of overlapped clusters.
            Then, the function calculates the mosaic image of each cluster.
            The generated mosaic images are then collected and saved to disk.


[1.6] depthmap.py
---------------
An auxiliary module for the serial program. It include only one function:
        
        [a] gen_mosaic(frames, cams, depth): This function receives the
            cameras, frames, depths of a certain cluster and calculates
            the sweep planes homographies. It then calculates the blended image
            and cost info. for each plane and end by applying graph-cut
            optimization to generate the output mosaic which is returned to the
            caller.

[1.7] mpi_controller.py
---------------
The starting module for the parallel program. It include only one starting function:
        
        [a] mosaic_controller(): This function defines the global master
            process and splits the world communicator into a set of virtual
            communicators, each responsible for a group of processes. The
            global master process then, loads the input frames and their cameras
            from disk and clusters them into a set of overlapped clusters.
            The clusters are grouped into different blocks based on the
            number of available virtual communicators. The global master process
            then, sends each block to the local master process of a virtual
            communicator responsible to perform the parallel Plane Sweep Stereo
            (PSS) algorithm. The local master process loops on the clusters of
            the assigned block and apply the parallel PSS to generate a mosaic
            image for each cluster. When a local master process finishes the
            assigned block, it sends the generated mosaics back to the global
            master process which accumulates all the generated mosaics and
            save them to disk.

[1.8] mpi_depthmap.py
---------------
An auxiliary module for the parallel program. It include only one function:
        
        [a] gen_mosaic(comm, in_frames, in_cams, in_depths): This function
            receives a virtual communicator, a set of input frames and cameras
            of a certain trajectory cluster. The local master process of the
            given communicator (process 0) calculates the homographies of all
            sweep planes  and broadcasts the received frames to all other
            processes. Block decomposition is used to partition the generation
            of sweep planes between the processes and and Scatterv is used to
            scatter the plane_homographies to all processes. Processes receive
            the homographies and perform gen_sweep_plane() on each assigned plane.
            The local master process then gather the costs and blended images, and
            apply graph-cut on them to output the interpolated mosaic which
            is returned back and accumulated by the local master process.



[1.9] util.py
---------------
A utility module that contain general purpose shared classes and functions:
    
    [a] Point class: Representing a 2D point
    
    [b] Rect class: Representing a rectangle
    
    [c] BLOCK_LO, BLOCK_HI, BLOCK_Size, BLOCK_OWNER: functions used by MPI for
        block decomposition.

[1.10] unit_test.py
---------------
A module that implements various tests for the functions in all project modules.

[2] 3rd Party Libraries
--------------------
The program uses the following libraries:
    
    [a] (cv2.so), Opencv (http://opencv.willowgarage.com/wiki/): on Centos,
        the supported version is 2.4.0 and on ubuntu, the supported version
        is 2.4.2. (Please look at Using the Makefile)
    
    [b] (Polygon folder), GPC – General Polygon Clipper library
        (http://www.cs.man.ac.uk/~toby/gpc/): for handling polygon operations.
        The library is compiled for both ubuntu and centos
    
    [c] (pygco.so), The gco library (http://vision.csd.uwo.ca/code/): used for
        multi-label graph-cut optimization. The library is compiled for
        both ubuntu and centos

[3] Using the Makefile:
--------------------------
The code comes in two versions to run on ubuntu and centos. Also, the makefile
has all the program parameters defined in top of the file for any changes.

The program assumes the input files in the data folder and outputs the generated
mosaics in the mosaics folder.

[3.1] For ubuntu:
---------------
The program assumes that opencv 2.4.2 is installed. If not then you should run,
"make install-opencv" first before running the code. The code can run in parallel
or serial mode using "make serial" or "make parallel".

[3.1.1] Rules:
---------------

    [a] parallel: will test the MPI implementation
    
    [b] serial: will test the serial implementation
    
    [c] install-opencv: will install the opencv libraries
    
    [d] clean: Cleans the output files in the mosaics/ folder

[3.2] For centos:

The program is compiled to be consistent with the centos operating system used in
the mako sharcnet cluster. the make file uses a run script to automatically load
the required modules. So, a user can test the program in parallel
or serial modes using "make serial" or "make parallel".

For running the code on mako sharcnet cluster, you need to copy all the contents
of the centos/ folder to a sharcnet folder. Then, run the code using the makefile.
It is preferable to run the program in mako cluster as you will not need
to install the Opencv libraries locally.

[3.2.1] Rules:
---------------

    [a] parallel: will test the MPI implementation
    
    [b] serial: will test the serial implementation
    
    [c] clean: Cleans the output files in the mosaics/ folder


[4] References
---------------

(1) M. A. Helala, L. A. Zarrabeitia, and F. Z. Qureshi, Mosaic of near ground
uav videos under parallax effects," in Proc. 6th ACM/IEEE International
Conference on Distributed Smart Cameras (ICDSC), Hong Kong, China, Oct. 2012.

[5] Contact
-----------------------------------------------------------------------------
Mohamed Helala,

Computer Science Deptartment, UOIT,

Oshawa, Ontario, Canada

mohamed.helala@uoit.ca             

http://vclab.science.uoit.ca/~helala/.
