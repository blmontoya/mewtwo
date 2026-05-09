from torch.utils.data import Dataset
import numpy as np
import torch
from pathlib import Path 
import os

import hyperparameters as hp
import params as p
from helpers import get_point_cloud, uv_proj
import yaml

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
            u -- (N,) u coordinate for each 3D pixel
            v -- (N,) v coordinate for each 3D pixel
            ranges -- (N,) ranges for each 3D pixel 

        """
        # 1. Get the scan at the given index and extract the point cloud from it
        scan_path = self.files[idx]
        
        #for later reprojection formatting, save parts of the file name
        scan_name = os.path.basename(scan_path).replace("bin", "")
        seq = str(scan_path).split("sequences/")[1].split("/")[0]

        label_path = Path(str(scan_path).replace("velodyne", "labels").replace(".bin", ".label"))
        raw_labels = np.fromfile(label_path, dtype=np.uint32).reshape((-1))

        instance_id = raw_labels >> 16
        raw_labels = raw_labels & 0xFFFF
        #convert the labels from random nums to 0-33 
        with open("config/semantic-kitti.yaml", 'r') as f: 
            DATA = yaml.safe_load(f)
        
        lookup = DATA["learning_map"]
        get_hash = np.vectorize(lookup.get, otypes=[int])
        labels = get_hash(raw_labels)
        
        xyz, remission = get_point_cloud(scan_path)
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

        # 2. Get new coordinates (u, v) 
        r = np.linalg.norm(xyz, axis=1)
        u, v = uv_proj(x, y, z, self.f_down, self.f, r, self.W, self.H)
        u = np.clip(u, 0, self.W - 1)
        v = np.clip(v, 0, self.H - 1)
        #print("u min/max:", u.min(), u.max()) #want in [0, W-1]
        #print("v min/max:", v.min(), v.max()) #want in [0, H-1]

        ranges = np.zeros((self.H, self.W), dtype=np.float32)
        xyz_proj = np.zeros((self.H, self.W, 3), dtype=np.float32)
        remission_proj = np.zeros((self.H, self.W), dtype=np.float32)
        label_proj = np.zeros((self.H, self.W), dtype=np.uint32)
        mask = np.zeros((self.H, self.W), dtype=np.float32)
        proj_idx = -np.ones((self.H, self.W), dtype=np.int32) #init as -1 to not confuse with actual indecies
        
        # 3. Assign points in descending range order (closer points will override further ones)
        order = np.argsort(r)[::-1] 

        for i in order:
            row, col = v[i], u[i]

            ranges[row, col] = r[i]
            xyz_proj[row, col] = xyz[i]
            remission_proj[row, col] = remission[i]
            label_proj[row,col] = labels[i]
            mask[row, col] = 1.0 
            proj_idx[row, col] = i

        # 4. Concatenate range, x, y, z, and remission to construct pts  
        pts = np.concatenate([
            ranges[..., None], 
            xyz_proj,
            remission_proj[..., None]
        ], axis=-1)
        
        input_img = torch.from_numpy(pts).permute(2, 0, 1).float()
        input_labels = torch.from_numpy(label_proj)
        mask = torch.from_numpy(mask).float()

        u = np.array(u)
        v = np.array(v)
        return (input_img, input_labels, mask, u, v, scan_name, seq, r)

    def get_splits(train, val, test):
        return {
            "train": train,
            "val": val,
            "test": test
        }