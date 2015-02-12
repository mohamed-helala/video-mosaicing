# Make file:
    # parallel rule: executes the parallel code using MPI
    # serial rule: executes the serial code
    # install-opencv rule: install opencv2.4.2
    # clean: removes the output mosaics

data_dir=./data # Data folder containing the frames and camera parameters file
P=20 # Number of processes
cP=4 # Number of processes used to generate a single cluster
cSize=6 # Number of frames per cluster
dStart=34 # Start of the depth range
dEnd=46  # End of the depth range
dStep=0.5 # depth step

parallel:
	mpiexec -n $(P) python mpi_controller.py  $(data_dir) $(cP) $(cSize) $(dStart) $(dEnd) $(dStep)

serial:
	python serial_controller.py $(data_dir) $(cSize) $(dStart) $(dEnd) $(dStep)

install-opencv:
	./scripts/opencv2.4.2-install.sh

clean:
	rm mosaics/*
