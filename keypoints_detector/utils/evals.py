"""


## Evaluate the performance in (x,y) coordinate
### Transform heatmap back to the (x,y) coordinate
In order to evaluate the model performance in terms of the root mean square error (RMSE) on (x,y) coordiantes,
I need to transform heatmap of the landmarks back to the (x,y) coordiantes.

The simplest way would be to use the (x,y) coordiantes of the pixcel with the largest estimated density as
the estimated coordinate. In this procedure, however, the estimated (x,y) coordinates are always integers
while the true (x,y) coordinates are not necessarily integers. Instead, we may use weighted average of the
(x,y) coordinates corresponding to the pixcels with the top "n_points" largest estimated density.

Then question is, how many "n_points" should we use to calculate the weighted average of the coordinates.
In the following script, I experiment the effects of the change in "n_points" on the RMSE using training
set For the choice of "n_points" I only consider n2 for integer n to allow that selected coordinates to
form symetric geometry.

RMSE is calculated in three ways:
- RMSE1: (x,y) coordinates from estimated heatmap VS (x,y) coordinates from true heatmap
- RMSE2: (x,y) coordinates from est heatmap VS true (x,y) coordinates of the landmark
- RMSE3: (x,y) coordinates from true heatmap VS true (x,y) coordinates of the landmark


Ideally, we want to find n_points that returns the smallest RMSE1, RMSE2 and RMSE3.

To reduce the computation time, I will only use the subset of training images for this experiment.

#### Results
The largest "n_points" = 96x96 does not return the smallest RMSE1, RMSE2 or RMSE3. This makes sense because
the iamge size is finite and some of the landmarks are at the corner. Taking weighted average across the entire
image may results will bias in the coordinate values toward the center of images. This would be the reason why
RMSE3 is never 0.

This observation makes me think that it would be better to make the n_points depends on each (image, landmark)
combination separately. For example, if the highest density point is at (0,0), then we should not consider large
n_points because the estimated density at (-1,-1) would have been large but density in such coordinates are not
estimated. On the other hand, if the highest density point is at around the center of the image, then I should
consider large n_points. For the simpliciy, I will not implement such procedure, and this would be my future work.

I will use n_points = 25 as it yields the smallest RMSE3

"""
import numpy as np
import matplotlib.pyplot as plt


def get_ave_xy(hmi, n_points=4, thresh=0):
    '''
    hmi      : heatmap np array of size (height,width)
    n_points : x,y coordinates corresponding to the top  densities to calculate average (x,y) coordinates


    convert heatmap to (x,y) coordinate
    x,y coordinates corresponding to the top  densities 
    are used to calculate weighted average of (x,y) coordinates
    the weights are used using heatmap

    if the heatmap does not contain the probability > 
    then we assume there is no predicted landmark, and 
    x = -1 and y = -1 are recorded as predicted landmark.
    '''
    if n_points < 1:
        # Use all
        hsum, n_points = np.sum(hmi), len(hmi.flatten())
        ind_hmi = np.array([range(input_width)] * input_height)
        i1 = np.sum(ind_hmi * hmi) / hsum
        ind_hmi = np.array([range(input_height)] * input_width).T
        i0 = np.sum(ind_hmi * hmi) / hsum
    else:
        ind = hmi.argsort(axis=None)[-n_points:]  # pick the largest n_points
        topind = np.unravel_index(ind, hmi.shape)
        index = np.unravel_index(hmi.argmax(), hmi.shape)
        i0, i1, hsum = 0, 0, 0
        for ind in zip(topind[0], topind[1]):
            h = hmi[ind[0], ind[1]]
            hsum += h
            i0 += ind[0] * h
            i1 += ind[1] * h

        i0 /= hsum
        i1 /= hsum
    if hsum / n_points <= thresh:
        i0, i1 = -1, -1
    return([i1, i0])


def transfer_xy_coord(hm, n_points=64, thresh=0.2):
    '''
    hm : np.array of shape (height,width, n-heatmap)

    transfer heatmap to (x,y) coordinates

    the output contains np.array (Nlandmark * 2,) 
    * 2 for x and y coordinates, containing the landmark location.
    '''
    assert len(hm.shape) == 3
    Nlandmark = hm.shape[-1]
    # est_xy = -1*np.ones(shape = (Nlandmark, 2))
    est_xy = []
    for i in range(Nlandmark):
        hmi = hm[:, :, i]
        est_xy.extend(get_ave_xy(hmi, n_points, thresh))
    return(est_xy)  # (Nlandmark * 2,)


def transfer_target(y_pred, thresh=0, n_points=64):
    '''
    y_pred : np.array of the shape (N, height, width, Nlandmark)

    output : (N, Nlandmark * 2)
    '''
    y_pred_xy = []
    for i in range(y_pred.shape[0]):
        hm = y_pred[i]
        y_pred_xy.append(transfer_xy_coord(hm, n_points, thresh))
    return(np.array(y_pred_xy))


def getRMSE(y_pred_xy, y_train_xy, pick_not_NA):
    res = y_pred_xy[pick_not_NA] - y_train_xy[pick_not_NA]
    RMSE = np.sqrt(np.mean(res**2))
    return(RMSE)


nimage = 500

rmelabels = ["(x,y) from est heatmap  VS (x,y) from true heatmap",
             "(x,y) from est heatmap  VS true (x,y)             ",
             "(x,y) from true heatmap VS true (x,y)             "]
n_points_width = range(1, 10)
res = []
n_points_final, min_rmse = -1, np.Inf
for nw in n_points_width + [0]:
    n_points = nw * nw
    y_pred_xy = transfer_target(y_pred[:nimage], 0, n_points)
    y_train_xy = transfer_target(y_tra[:nimage], 0, n_points)
    pick_not_NA = (y_train_xy != -1)

    ts = [getRMSE(y_pred_xy, y_train_xy, pick_not_NA)]
    ts.append(getRMSE(y_pred_xy, y_train0.values[:nimage], pick_not_NA))
    ts.append(getRMSE(y_train_xy, y_train0.values[:nimage], pick_not_NA))

    res.append(ts)

    print("n_points to evaluate (x,y) coordinates = {}".format(n_points))
    print(" RMSE")
    for r, lab in zip(ts, rmelabels):
        print("  {}:{:5.3f}".format(lab, r))

    if min_rmse > ts[2]:
        min_rmse = ts[2]
        n_points_final = n_points

res = np.array(res)
for i, lab in enumerate(rmelabels):
    plt.plot(n_points_width + [input_width], res[:, i], label=lab)
plt.legend()
plt.ylabel("RMSE")
plt.xlabel("n_points")
plt.show()

