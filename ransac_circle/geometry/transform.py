import numpy as np
# ---------------------------------------------------------------
# 3D transformation (used on whole point cloud)
# ---------------------------------------------------------------
def tr_matrix(dx, dy, dz, a1, a2, a3):
    a1, a2, a3 = np.radians([a1, a2, a3])
    T = np.array([[1,0,0,0],
                  [0,1,0,0],
                  [0,0,1,0],
                  [-dx,-dy,-dz,1]], dtype=float)

    R1 = np.array([[1,0,0,0],
                   [0,np.cos(a1),-np.sin(a1),0],
                   [0,np.sin(a1), np.cos(a1),0],
                   [0,0,0,1]], dtype=float)

    R2 = np.array([[ np.cos(a2),0,np.sin(a2),0],
                   [ 0,1,0,0],
                   [-np.sin(a2),0,np.cos(a2),0],
                   [ 0,0,0,1]], dtype=float)

    R3 = np.array([[np.cos(a3), -np.sin(a3),0,0],
                   [np.sin(a3),  np.cos(a3),0,0],
                   [0,0,1,0],
                   [0,0,0,1]], dtype=float)

    return T @ R3 @ R2 @ R1