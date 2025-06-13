import os
import numpy as np
from common_src.model.utils import LiDARInstance3DBoxes

import torch
from torch.utils.data import Dataset

from vod.configuration import KittiLocations
from vod.frame import FrameDataLoader, FrameTransformMatrix, homogeneous_transformation

from common_src.augment.augmentation import DataAugmenter, load_kitti_calib
from common_src.augment.utils import apply_global_transformations
from pathlib import Path

class ViewOfDelft(Dataset):
    CLASSES = [
        "Car",
        "Pedestrian",
        "Cyclist",
    ]
    #    'rider',
    #    'unused_bicycle',
    #    'bicycle_rack',
    #    'human_depiction',
    #    'moped_or_scooter',
    #    'motor',
    #    'truck',
    #    'other_ride',
    #    'other_vehicle',
    #    'uncertain_ride'

    LABEL_MAPPING = {
        "class": 0,  # Describes the type of object: 'Car', 'Pedestrian', 'Cyclist', etc.
        "truncated": 1,  # Not used, only there to be compatible with KITTI format.
        "occluded": 2,  # Integer (0,1,2) indicating occlusion state 0 = fully visible, 1 = partly occluded 2 = largely occluded.
        "alpha": 3,  # Observation angle of object, ranging [-pi..pi]
        "bbox2d": slice(4, 8),
        "bbox3d_dimensions": slice(
            8, 11
        ),  # 3D object dimensions: height, width, length (in meters).
        "bbox3d_location": slice(
            11, 14
        ),  # 3D object location x,y,z in camera coordinates (in meters).
        "bbox3d_rotation": 14,  # Rotation around -Z-axis in LiDAR coordinates [-pi..pi].
    }
    
    def __init__(self, 
                 data_root = 'data/view_of_delft', 
                 sequential_loading=False,
                 split = 'train',
                 augmentation_cfg=None):
        super().__init__()

        self.data_root = data_root
        assert split in [
            "train",
            "val",
            "test",
        ], f"Invalid split: {split}. Must be one of ['train', 'val', 'test']"
        self.split = split
        split_file = os.path.join(data_root, "lidar", "ImageSets", f"{split}.txt")

        with open(split_file, "r") as f:
            lines = f.readlines()
            self.sample_list = [line.strip() for line in lines]
        
        self.vod_kitti_locations = KittiLocations(root_dir = data_root)

        self.augmentation_cfg = augmentation_cfg
        self.augmenter = None
        self.global_augment_cfg = None

        if augmentation_cfg and augmentation_cfg.enabled and self.split == 'train':
            if augmentation_cfg.copy_paste.enabled:
                self.augmenter = DataAugmenter(cfg=augmentation_cfg.copy_paste)
            if augmentation_cfg.global_transforms.enabled:
                self.global_augment_cfg = augmentation_cfg.global_transforms 


    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        num_frame = self.sample_list[idx]
        vod_frame_data = FrameDataLoader(
            kitti_locations=self.vod_kitti_locations, frame_number=num_frame
        )
        local_transforms = FrameTransformMatrix(vod_frame_data)

        lidar_data = vod_frame_data.lidar_data

        # Apply object augmentation and insert
        if self.augmenter: 
            calib_dir = self.vod_kitti_locations.lidar_calib_dir
            calib_path = os.path.join(calib_dir, f"{num_frame}.txt")
            calib_dict = load_kitti_calib(Path(calib_path))
            data_sample = {
                'pc': vod_frame_data.lidar_data.copy(),
                'labels': vod_frame_data.raw_labels.copy(),
                'calib': calib_dict
            }
            augmented_sample, _ = self.augmenter(data_sample)
            lidar_data = augmented_sample['pc']
            raw_labels = augmented_sample['labels']
        else:
            lidar_data = vod_frame_data.lidar_data
            raw_labels = vod_frame_data.raw_labels

        lidar_data = lidar_data[:, :4]

        gt_labels_3d_list = []
        gt_bboxes_3d_list = []
        if self.split != 'test':
            # raw_labels = vod_frame_data.raw_labels // check if this is a bug later
            for idx, label in enumerate(raw_labels):
                label = label.split(" ")

                if label[self.LABEL_MAPPING["class"]] in self.CLASSES:

                    gt_labels_3d_list.append(
                        int(self.CLASSES.index(label[self.LABEL_MAPPING["class"]]))
                    )

                    bbox3d_loc_camera = np.array(
                        label[self.LABEL_MAPPING["bbox3d_location"]]
                    )
                    trans_homo_cam = np.ones((1, 4))
                    trans_homo_cam[:, :3] = bbox3d_loc_camera
                    bbox3d_loc_lidar = homogeneous_transformation(trans_homo_cam, local_transforms.t_lidar_camera)
                    
                    bbox3d_locs = np.array(bbox3d_loc_lidar[0,:3], dtype=np.float32)         
                    bbox3d_dims = np.array(label[self.LABEL_MAPPING['bbox3d_dimensions']], dtype=np.float32)[[2, 1, 0]] # hwl -> lwh
                    bbox3d_rot = np.array([label[self.LABEL_MAPPING['bbox3d_rotation']]], dtype=np.float32)
                
                    gt_bboxes_3d_list.append(np.concatenate([bbox3d_locs, bbox3d_dims, bbox3d_rot], axis=0))
        
        if not gt_bboxes_3d_list:
            gt_labels_3d_np = np.array([], dtype=np.int64)
            gt_bboxes_3d_np = np.zeros((0, 7), dtype=np.float32)
        else:
            gt_labels_3d_np = np.array(gt_labels_3d_list, dtype=np.int64)
            gt_bboxes_3d_np = np.stack(gt_bboxes_3d_list, axis=0)

        if self.global_augment_cfg:
            scaling = np.random.uniform(*self.global_augment_cfg.scaling_range)
            rotation = np.random.uniform(*self.global_augment_cfg.rotation_range)
            lidar_data, gt_bboxes_3d_np = apply_global_transformations(
                lidar_data, gt_bboxes_3d_np, rotation, scaling
            )

        gt_labels_3d = torch.from_numpy(gt_labels_3d_np)
        gt_bboxes_3d = torch.from_numpy(gt_bboxes_3d_np)
        
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0))
        
        lidar_data = torch.tensor(lidar_data)
        
        return dict(
            lidar_data=lidar_data,
            image_data=image_data,
            transforms=local_transforms,
            gt_labels_3d=gt_labels_3d,
            gt_bboxes_3d=gt_bboxes_3d,
            meta=dict(num_frame=num_frame),
        )
