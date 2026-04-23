import numpy as np
import math

def eval_inliers_fast(x_data, y_data, xc, yc, R_sq, thresh):
    n = x_data.shape[0]
    out = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        dx = x_data[i] - xc
        dy = y_data[i] - yc
        d2 = dx*dx + dy*dy
        if abs(d2 - R_sq) < thresh:
            out[i] = True
    return out
def eval_inliers_fast(x_data, y_data, xc, yc, R_sq, thresh):
    dx = x_data - xc
    dy = y_data - yc
    d2 = dx*dx + dy*dy
    return np.abs(d2 - R_sq) < thresh

