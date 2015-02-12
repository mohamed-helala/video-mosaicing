from pygco import *
from util import *


# This function takes a list of grid arrays, one for each label. Each array
# represents the costs of its assigned label. The given list has the structure
# [nLabels, w, h]. It also takes a rec for each grid array representing its 2D
# position with respect to other grids (This is too important for aligning
# costs).


def gen_gco_labels(bldImages, costs, maxrecs):

    # get max rec
    max_poly = maxrecs[0].get_poly()

    for k in range(len(maxrecs) - 1):
        max_poly = max_poly | maxrecs[k + 1].get_poly()
    #end for

    (xmin, xmax, ymin, ymax) = max_poly.boundingBox(0)
    max_rec = Rect(Point(xmin, ymin), Point(xmax, ymax))

    nLabels = len(costs)
    w = int(max_rec.width())
    h = int(max_rec.height())

    cost_data = np.empty([w, h, nLabels], dtype=np.int32)
    cost_data[:w, :h, :nLabels] = 1000
    for k in range(nLabels):
        dx = int(np.round(maxrecs[k].left - max_rec.left))
        dy = int(np.round(maxrecs[k].top - max_rec.top))
	rec_h = maxrecs[k].height()
	rec_w = maxrecs[k].width()
        cost_data[dx:dx + rec_w, dy:dy + rec_h, k] = costs[k][:, :]
    #end for

    labels = optMosaic(cost_data, 5, 50, 2)
    # copy the pixels corresponding to each label from the blend images
    op_im = np.zeros([h, w, 3], dtype=np.uint8)
    for x in range(w):
        for y in range(h):
            l = labels[x, y]
            dx = int(np.round(maxrecs[l].left - max_rec.left))
            dy = int(np.round(maxrecs[l].top - max_rec.top))
            op_im[y, x, :3] = bldImages[l][y - dy, x - dx, :3]
        #end for
    #end for

    return op_im, labels, max_rec
