import numpy as np


class CamInfo(object):  # This class holds the camera information
    def __init__(self, K=None,
                    E=None,
                    P=None,
                    r=None,
                    t=None):
            self.K = K
            self.P = P
            self.E = E
            self.r = r
            self.t = t

    def linear_size(self):
        return np.prod(self.K.shape) + np.prod(self.P.shape) \
                + np.prod(self.E.shape) + np.prod(self.r.shape) \
                + np.prod(self.t.shape)

    def linearize(self):
        lin = np.concatenate((self.K.ravel(), self.P.ravel()))
        lin = np.concatenate((lin, self.E.ravel()))
        lin = np.concatenate((lin, self.r.ravel()))
        lin = np.concatenate((lin, self.t.ravel()))
        return lin

    def create_from_linear(self, lin_arr, shapes):
        disp = 0
        idxs = []
        for k in range(len(shapes) - 1):
            disp += np.prod(shapes[k])
            idxs.append(disp)
        #end for
        arr = np.split(lin_arr, idxs)

        self.K = arr[0]
        self.K.shape = shapes[0]
        self.P = arr[1]
        self.P.shape = shapes[1]
        self.E = arr[2]
        self.E.shape = shapes[2]
        self.r = arr[3]
        self.r.shape = shapes[3]
        self.t = arr[4]
        self.t.shape = shapes[4]

        return self


def adjust_cams(v_cam, src_cams):
    dst_cams = []
    Rv = v_cam.E[:3, :3].copy()  # Get copy of the rotation matrix
    tv = v_cam.E[:3, 3].copy()  # Get copy of the translation vector
    Rv = np.transpose(Rv)
    for cam in src_cams:
        new_cam = CamInfo(cam.K.copy(), cam.E.copy(),
                          cam.P.copy(), cam.r.copy(), cam.t.copy())
        Rs = new_cam.E[:3, :3]
        ts = new_cam.E[:3, 3]
        Rs[:3, :3] = np.dot(Rs, Rv)  # inplace modification of Rs
        ts[:3] = ts - np.dot(Rs, tv)  # inplace modification of ts
        dst_cams.append(new_cam)
    #end for
    return dst_cams

