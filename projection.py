from torch.utils.data import Dataset
import numpy as np
import torch
from pathlib import Path 

import hyperparameters as hp
import params as p
from helpers import get_point_cloud, uv_proj

class RangeData(Dataset): 
    """Create a dataset of range images from Semantic KITTI point clouds

    Arguments:
        xyz:        (x,y,z) coordinates of point cloud (N, 3)

    Returns:
        uv          (u,v) range image tensor
    """
    def __init__(self, root, sequences, H=hp.H_RANGE, W=hp.W_RANGE):
        self.root = Path(root)
        self.H = H
        self.W = W

        self.files = []
        for seq in sequences: 
            seq = f"{int(seq):02d}"
            self.files += sorted((self.root / "sequences" / seq / "velodyne").glob("*.bin"))

        self.f_up = np.deg2rad(p.F_UP)
        self.f_down = np.deg2rad(p.F_DOWN)
        self.f = np.abs(self.f_up) + np.abs(self.f_down)

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx): 
        """Return a range image and projected label for a given input scan index

        Returns:
            image  -- (5, H, W) representing range, x, y, z, and remission for each
            pojected coordinate (u,v)
            label -- (H, W) representing label for each projected coordinate (u,v)
            mask -- (H, W) set to "0" if the pixel is empty 

        """
        
        # 1. Get the scan at the given index and extract the point cloud from it
        scan_path = self.files[idx]
        label_path = Path(str(scan_path).replace("velodyne", "labels").replace(".bin", ".label"))
        labels = np.fromfile(label_path, dtype=np.uint32).reshape((-1))
        
        xyz, remission = get_point_cloud(scan_path)
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

        # 2. Get new coordinates (u, v) 
        r = np.linalg.norm(xyz, axis=1)
        u, v = uv_proj(x, y, z, self.f_up, self.f, r, self.W, self.W)

        ranges = np.zeros((self.H, self.W), dtype=np.float32)
        xyz_proj = np.zeros((self.H, self.W, 3), dtype=np.float32)
        remission_proj = np.zeros((self.H, self.W), dtype=np.float32)
        label_proj = np.zeros((self.H, self.W), dtype=np.uint32)
        mask = np.zeros((self.H, self.W), dtype=np.float32)
        
        # 3. Assign points in descending range order (closer points will override further ones)
        order = np.argsort(r)[::-1] 

        for i in order: 
            row, col = u[i], v[i]

            ranges[row, col] = r[i]
            xyz_proj[row, col] = xyz[i]
            remission_proj[row, col] = remission[i]
            label_proj[row,col] = labels[i]
            mask[row, col] = 1.0 

        # 4. Concatenate range, x, y, z, and remission to construct pts  
        pts = np.concatenate([
            ranges[..., None], 
            xyz_proj,
            remission_proj[..., None]
        ])

        input_img = torch.from_numpy(pts).permute(2, 0, 1).float()
        input_labels = torch.from_numpy(label_proj)
        mask = torch.from_numpy(mask).float()

        return (input_img, input_labels, mask)

