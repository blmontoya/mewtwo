# postprocess.py
import numpy as np
from sklearn.neighbors import KDTree

def postprocess(xyz, range_preds, u, v, k=5):
    """
    Runs KNN voting in 3D to clean up label bleeding.

    Args:
        xyz:          (N, 3)  original 3D point positions
        range_preds:  (H, W)  predicted class labels on range image
        u, v:         (N,)    pixel coords each point projected to
        k:            number of neighbors for voting

    Returns:
        final_preds:  (N,)  refined per-point class labels
    """
    # pull initial per-point labels straight from the range image
    point_preds = range_preds[v, u].astype(np.int32)

    # KDTree over xyz for fast neighbor lookup
    tree = KDTree(xyz)
    _, idxs = tree.query(xyz, k=k)

    # majority vote among each point's k neighbors
    neighbor_labels = point_preds[idxs]
    final_preds = np.array([
        np.bincount(row).argmax()
        for row in neighbor_labels
    ])
    return final_preds