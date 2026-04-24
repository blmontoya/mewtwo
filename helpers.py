import numpy as np 
import hyperparameters as hp
import params as p

# Projection Helpers
def get_point_cloud(file_name): 
    scan = np.fromfile(file_name, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    xyz = scan[:, :3] 
    resmission = scan[:, 3]
    return xyz, resmission 

def uv_proj(x, y, z, f_down, f, r, w, h):
    u = 0.5 * (1 - np.arctan2(y,x) / np.pi) * w
    v = (1 - ((np.arcsin(z / r) + np.abs(f_down)) / f)) * h

    u = np.floor(u).astype(np.int32)
    v = np.floor(v).astype(np.int32)

    return u, v

# ========================================================================