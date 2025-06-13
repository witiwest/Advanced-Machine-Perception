from pathlib import Path
from collections import namedtuple
import numpy as np
import random
import pickle
import os
import glob

Box = namedtuple("Box", "x y z l w h yaw")

def fit_global_ground_plane_ransac(pc: np.ndarray, iters=50, eps=0.1):
    """
    Fits a single, global ground plane to the entire scene using RANSAC.

    It focuses on points near the ego-vehicle to get a stable estimate, assuming
    this represents the primary ground plane.
    """
    # First, filter the point cloud to a "trusted" subset for finding the ground.
    # We use points that are relatively close to the vehicle and below the sensor height.
    # RANSAC Implementation 
    best_inliers_count = 0
    best_plane = None
    rng = np.random.default_rng()
    
    trusted_mask = (np.linalg.norm(pc[:, :2], axis=1) < 20.0) & (pc[:, 2] < -0.5)
    ground_candidates = pc[trusted_mask, :3]


    if len(ground_candidates) < 50:
        print("Warning: Not enough ground points near vehicle to fit a global plane.")
        return None

    for _ in range(iters):
        try:
            p1, p2, p3 = ground_candidates[rng.choice(len(ground_candidates), 3, replace=False)]
        except ValueError:
            continue

        normal = np.cross(p2 - p1, p3 - p1)
        norm_val = np.linalg.norm(normal)
        if norm_val < 1e-6: continue

        # Ensure the plane normal points generally upwards
        if normal[2] < 0: normal *= -1.0
        normal /= norm_val

        # Reject planes that are too vertical
        if normal[2] < 0.85: continue
        
        d = -normal.dot(p1)
        distances = np.abs(ground_candidates @ normal + d)
        current_inliers_count = np.sum(distances < eps)

        if current_inliers_count > best_inliers_count:
            best_inliers_count = current_inliers_count
            best_plane = (*normal, d)

    return best_plane

def apply_noise_to_object(points, rng, jitter_std=0.02, dropout_prob=0.05):
    """
    Applies random noise to the points of a donor object.
    """
    # Point Jitter: add small random values to each point's XYZ coordinates.
    # We create noise with the same shape as the XYZ data and a given std. dev.
    jitter = rng.normal(scale=jitter_std, size=(points.shape[0], 3))
    points[:, :3] += jitter # Add the noise to the X, Y, Z columns

    # Point Dropout: randomly remove a small percentage of points.
    if dropout_prob > 0:
        # Create a random mask. Points are kept if their random value is > dropout_prob.
        keep_mask = rng.random(size=len(points)) > dropout_prob
        points = points[keep_mask]

    return points

def _corners_2d(b: Box):
    """ Helper for SAT: gets the 2D corners of a bounding box. """
    dx, dy = b.l / 2, b.w / 2
    c, s   = np.cos(b.yaw), np.sin(b.yaw)
    R      = np.array([[c, -s], [s, c]])
    local  = np.array([[dx, dy], [-dx, dy], [-dx, -dy], [dx, -dy]])
    return (R @ local.T).T + np.array([b.x, b.y])

def _overlap_bev(a: Box, b: Box, margin=0.1):
    """ Helper for SAT: performs 2D Separating Axis Theorem check. """
    A, B = _corners_2d(a), _corners_2d(b)
    axes = np.vstack([A[1] - A[0], A[0] - A[3], B[1] - B[0], B[0] - B[3]])
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    for ax in axes:
        pA, pB = A @ ax, B @ ax
        if pA.max() + margin < pB.min() or pB.max() + margin < pA.min():
            return False
    return True

def boxes_overlap(a: Box, b: Box, m_xy=0.01, m_z=0.01):
    """ Checks for 3D overlap between two boxes using BEV SAT + Z-axis check. """
    z_ok = abs(a.z - b.z) < (a.h + b.h) / 2 + m_z
    return z_ok and _overlap_bev(a, b, m_xy)


    
def build_voxel_hash(points, voxel_size):
    """Returns a set of voxel keys occupied by points."""
    keys = np.floor(points / voxel_size).astype(np.int32)
    return {tuple(k) for k in keys}

def get_voxel_key(point, voxel_size):
    return tuple(np.floor(point / voxel_size).astype(np.int32))

# def build_depth_map(pc, n_az=360, n_el=64):
#     xyz    = pc[:, :3]
#     ranges = np.linalg.norm(xyz, axis=1)
#     az     = np.arctan2(xyz[:,1], xyz[:,0])
#     el     = np.arcsin(xyz[:,2] / ranges)

#     az_b = np.clip(((az + np.pi) / (2*np.pi) * n_az).astype(int), 0, n_az-1)
#     el_b = np.clip(((el + np.pi/2) / np.pi      * n_el).astype(int), 0, n_el-1)

#     depth_map = np.full((n_az, n_el), np.inf, dtype=np.float32)
#     for a, e, r in zip(az_b, el_b, ranges):
#         if r < depth_map[a, e]:
#             depth_map[a, e] = r

#     return depth_map

# def is_line_of_sight_clear_spherical(pc, obj_pts, depth_map, margin=0.25):
#     clear_mask = []
#     for pt in obj_pts:
#         r = np.linalg.norm(pt)
#         az = np.arctan2(pt[1], pt[0])
#         el = np.arcsin(pt[2] / r)
#         i  = int(((az + np.pi) / (2*np.pi) * depth_map.shape[0]))
#         j  = int(((el + np.pi/2) / np.pi      * depth_map.shape[1]))
#         i  = np.clip(i, 0, depth_map.shape[0]-1)
#         j  = np.clip(j, 0, depth_map.shape[1]-1)

#         clear_mask.append(r + margin < depth_map[i, j])

#     return obj_pts[np.array(clear_mask)]


def get_data_driven_sampler_pool(pc: np.ndarray, cls: str):
    """
    Filters the scene's point cloud to find all points that are likely on a
    traversable ground surface and respects class-specific sampling rules
    (e.g., for Car, Pedestrian, Cyclist).
    """
    # Basic ground filter: remove points too high or too close to the ego-vehicle
    mask = (pc[:,2] < -0.5) & (np.linalg.norm(pc[:,:2],axis=1)>1.0)
    x,y = pc[mask,:2].T
    if cls=="Car":
        region = (x>=7)&(x<=30)&(y>=-0.5*x)&(y<=0.5*x)
    elif cls=="Pedestrian":
        # allow x∈[0,20], y∈[-10,10]  (sidewalk / crosswalk region)
        region = (x>=2)&(x<=20)&(y>=-0.5*x)&(y<=0.5*x)
    else:  # Cyclist
        region = (x>=4)&(x<=30)&(np.abs(y)<=0.5*x)
    return pc[mask,:2][region]

def sample_pose_by_class(cls: str, rng, sampler_pool: np.ndarray):
    """
    Generates a realistic pose by sampling from a pool of valid ground points
    and applying class-specific rules.
    """
    # Randomly select a candidate (x,y) from the pre-filtered ground points.
    # This ensures we always start from a plausible location.
    if len(sampler_pool) == 0:
        return None, None, None 
    
    candidate_idx = rng.choice(len(sampler_pool))
    x, y = sampler_pool[candidate_idx]

    # Class-Specific heuristics and dead zones
    if cls == "Car":

        # Cars should have an orientation aligned with the road
        yaw = rng.normal(loc=0.0, scale=np.deg2rad(5))
        

    elif cls == "Pedestrian" or cls == "Cyclist":

        # Pedestrians can face any way, cyclists are mostly forward
        yaw = rng.uniform(-np.pi, np.pi) if cls == "Pedestrian" else rng.normal(loc=0.0, scale=np.deg2rad(20))

    return x, y, yaw

def is_placement_realistic(box: Box, cls: str, scene_pc: np.ndarray):
    """ Context-aware placement check using local ground geometry. """
    local_patch_radius = 1
    distances = np.linalg.norm(scene_pc[:, :2] - np.array([box.x, box.y]), axis=1)
    local_points = scene_pc[distances < local_patch_radius]
    if len(local_points) < 40: return False

    ground_points = local_points[local_points[:, 2] < (local_points[:, 2].min() + 0.25)]
    if len(ground_points) < 30: return False
    
    z_std_dev = np.std(ground_points[:, 2])

    if cls == "Car":
        # Cars must be on very flat surfaces
        return z_std_dev < 0.04  
    if cls in ["Pedestrian", "Cyclist"]:
        # Tighter threshold rejects vertical walls but allows for curbs
        return z_std_dev < 0.10 
    return True

def get_points_in_box(pc: np.ndarray, box: Box):
    """ Helper to get indices of points from a cloud inside a 3D box. """
    c, s = np.cos(-box.yaw), np.sin(-box.yaw)
    R_inv = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    translated_points = pc[:, :3] - np.array([box.x, box.y, box.z])
    rotated_points = (R_inv @ translated_points.T).T
    in_box_mask = (np.abs(rotated_points[:, 0]) < box.l / 2) & \
                  (np.abs(rotated_points[:, 1]) < box.w / 2) & \
                  (np.abs(rotated_points[:, 2]) < box.h / 2)
    return np.where(in_box_mask)[0]

def normalize_angles(angles: np.ndarray) -> np.ndarray:
    """
    Normalize angles to [0, 2*pi).
    """
    return np.mod(angles, 2 * np.pi)


def cross2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    2D cross product (result is scalar for each pair of vectors).
    a, b: (..., 2)
    returns: (...) cross product a.x*b.y - a.y*b.x
    """
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def remove_points_occluded_by_boxes(
    pc: np.ndarray,
    boxes: list,
    sensor_origin: np.ndarray = np.array([0.0, 0.0, 0.0]),
) -> np.ndarray:
    """
    Efficiently removes points from `pc` that are occluded by 3D boxes.
    """
    if not boxes:
        return pc

    pts = pc.copy()
    rel_xy = pts[:, :2] - sensor_origin[:2]
    ranges = np.linalg.norm(rel_xy, axis=1)
    angles = normalize_angles(np.arctan2(rel_xy[:,1], rel_xy[:,0]))

    occluded = np.zeros(len(pts), dtype=bool)

    for box in boxes:

        # Footprint corners and angular sector
        corners = _corners_2d(box)  # (4,2)
        c_rel = corners - sensor_origin[:2]
        c_angles = normalize_angles(np.arctan2(c_rel[:,1], c_rel[:,0]))
        a_min, a_max = c_angles.min(), c_angles.max()
        # handle wrap-around
        if a_max - a_min > np.pi:
            # shift small angles up
            c_angles = np.where(c_angles < (a_min + a_max)/2, c_angles + 2*np.pi, c_angles)
            a_min, a_max = c_angles.min(), c_angles.max()

        # select candidate rays
        angs = angles.copy()
        angs = np.where(angs < a_min, angs + 2*np.pi, angs)
        mask_sector = (angs >= a_min) & (angs <= a_max)  & ~occluded
        idxs = np.nonzero(mask_sector)[0]
        if idxs.size == 0:
            continue

        # Ray directions for candidates
        dirs = np.stack((np.cos(angs[idxs]), np.sin(angs[idxs])), axis=1)  # (K,2)
        origins = np.repeat(sensor_origin[:2][None, :], len(idxs), axis=0)  # (K,2)

        # Precompute edges
        pts2d = corners
        edge_starts = pts2d
        edge_ends = np.vstack((pts2d[1:], pts2d[0]))  # (4,2)
        e_vecs = edge_ends - edge_starts  # (4,2)

        # For each edge, compute t and u for all rays
        t_all = np.full((len(idxs), len(e_vecs)), np.inf)
        for j, (p1, v2) in enumerate(zip(edge_starts, e_vecs)):
            v1 = origins - p1  # (K,2)
            denom = cross2(dirs, v2)  # (K,)
            # avoid division by zero
            valid = np.abs(denom) > 1e-8
            t = np.full(len(idxs), np.inf)
            u = np.zeros(len(idxs))
            # compute t and u only where valid
            t_valid = cross2(v2, v1)[valid] / denom[valid]
            u_valid = cross2(dirs[valid], v1[valid]) / denom[valid]
            # accept intersections with t>0 and u in [0,1]
            mask_valid = (t_valid > 0) & (u_valid >= 0) & (u_valid <= 1)
            t[valid] = np.where(mask_valid, t_valid, np.inf)
            t_all[:, j] = t

        # nearest intersection per ray
        t_min = t_all.min(axis=1)
        # mark occluded if point distance > intersection
        occluded[idxs] = ranges[idxs] > t_min

    return pts[~occluded]


def _read_12(line):
    """Convert 'tag: 12 floats' into (3,4) array."""
    return np.array([float(x) for x in line.split()[1:]], dtype=np.float32).reshape(3, 4)

def load_kitti_calib(calib_path: Path):
    """
    Returns a dict {P2, Tr_velo_to_cam, R0_rect}.
    """
    with open(calib_path, "r") as f: lines = f.readlines()
    P2 = _read_12([l for l in lines if l.startswith("P2:")][0])
    Tr = _read_12([l for l in lines if l.startswith("Tr_velo_to_cam:")][0])
    r_line = [l for l in lines if l.startswith("R0_rect:")][0].split()[1:]
    R0 = np.array([float(x) for x in r_line], dtype=np.float32).reshape(3, 3)
    return {"P2": P2, "Tr_velo_to_cam": Tr, "R0_rect": R0}

# I/O helpers
def load_point_cloud(p: Path):
    return np.fromfile(p, np.float32).reshape(-1, 4)

def save_point_cloud(p, pts): 
    pts.astype(np.float32).tofile(p)

def load_labels(p):           
    return Path(p).read_text().splitlines()

def save_labels(p, lines):    
    Path(p).write_text("\n".join(l.rstrip() for l in lines)+"\n")


class DataAugmenter:
    def __init__(self, cfg):
        """
        Initializes the on-the-fly data augmenter. This is done ONCE per training run.
        """
        with open(cfg.obj_db_path, "rb") as f:
            self.obj_db = pickle.load(f)
        
        self.cfg = cfg
        self.augmentation_prob = self.cfg.prob
        self.max_trials = self.cfg.max_trials
        self.classes_to_augment = list(self.obj_db.keys())
        self.rng = np.random.default_rng()

        print(f"Data Augmenter initialized. Augmenting with: {self.classes_to_augment}")

    def __call__(self, data_sample: dict):
        if self.rng.random() > self.cfg.prob:
            return data_sample, []

        pc = data_sample['pc'].copy()
        labels = data_sample['labels'].copy()
        calib = data_sample['calib']
        
        # Pre computation for efficiency
        global_plane_params = fit_global_ground_plane_ransac(pc)
        scene_voxel_set = build_voxel_hash(pc[:, :3], voxel_size=0.1)
        
        Tr_cam_to_velo = np.linalg.inv(np.vstack([calib["Tr_velo_to_cam"], [0, 0, 0, 1]]))
        scene_boxes = []
        for ln in labels:
            p = ln.split(' ')
            if p[0] not in self.classes_to_augment: 
                continue
            
            h, w, l = map(float, p[8:11]); 
            x, y , z = map(float, p[11:14]); 
            ry = float(p[14])
            loc_velo = (Tr_cam_to_velo @ np.array([x, y , z , 1.0]))[:3]
            scene_boxes.append(Box(loc_velo[0], loc_velo[1], loc_velo[2], l , w , h, -(ry+np.pi/2)))

        # Multi-object insertion loop 
        successfully_placed_objects = []
        if self.cfg.multi_object.enabled:
            for i in range(self.cfg.multi_object.max_objects):
                if self.rng.random() > self.cfg.multi_object.attempt_probs[i]: 
                    break
                cls_to_insert = self.rng.choice(self.classes_to_augment)
                sampler_pool = get_data_driven_sampler_pool(pc, cls=cls_to_insert)
                # _perform_insertion gets the pre-computed voxel set
                new_object_data = self._perform_insertion(
                    pc, self.obj_db, cls_to_insert, scene_boxes, self.rng, 
                    global_plane_params, sampler_pool, scene_voxel_set, Tr_cam_to_velo)

                if new_object_data:
                    successfully_placed_objects.append(new_object_data)
                    scene_boxes.append(new_object_data['box'])
                else:
                    break
        
        else:
            # Single object insertion logic
            cls_to_insert = self.rng.choice(self.classes_to_augment)
            new_object_data = self._perform_insertion(
                pc, self.obj_db, cls_to_insert, scene_boxes, self.rng, 
                global_plane_params, sampler_pool, scene_voxel_set, Tr_cam_to_velo)
            if new_object_data:
                successfully_placed_objects.append(new_object_data)
        
        # Final scene composition
        if successfully_placed_objects:
            final_pc = pc.copy()
            final_boxes = [obj['box'] for obj in successfully_placed_objects]
            final_pc = remove_points_occluded_by_boxes(final_pc, final_boxes)

            if final_pc.shape[1] == 4:
                sem_base = np.zeros((final_pc.shape[0], 5), dtype=np.float32); 
                sem_base[:, 0] = 1.0
                final_pc = np.hstack([final_pc, sem_base])

            # Stack all the new object points and update labels
            for obj in successfully_placed_objects:
                final_pc = np.vstack([final_pc, obj['pts']])
                labels.append(obj['label'])
            
            data_sample['pc'] = final_pc
            data_sample['labels'] = labels

        return data_sample, successfully_placed_objects
    
    def _perform_insertion(self, original_pc, obj_db, cls, scene_boxes, rng, global_plane_params, sampler_pool, scene_voxel_set, Tr_cam_to_velo):
        
        if global_plane_params is None: 
            return None
        
        donors = obj_db.get(cls, [])
        if not donors: 
            return None
        
        for _ in range(self.cfg.max_trials):
            ent = rng.choice(donors); 
            label=ent["label"]
            pts = ent["points"].copy(); 

            cam_offset = np.array([float(label.split()[11]), float(label.split()[12]), float(label.split()[13]), 1.0])
            lidar_offset = (Tr_cam_to_velo@ cam_offset)
            x_offset_lidar = lidar_offset[0]
            y_offset_lidar = lidar_offset[1]

            pts[:, 0] -= x_offset_lidar
            pts[:, 1] -= y_offset_lidar

            original_ry_kitti = float(label.split()[14])
            original_yaw_velo = -(original_ry_kitti + np.pi / 2) 
            # original_yaw_velo = normalize_angles(original_yaw_velo + np.pi) - np.pi
            # Rz_original_inv = np.array([
            #     [np.cos(-original_yaw_velo), -np.sin(-original_yaw_velo), 0],
            #     [np.sin(-original_yaw_velo), np.cos(-original_yaw_velo), 0],
            #     [0, 0, 1]
            # ], np.float32)
            # pts[:,:3] = (Rz_original_inv @ pts[:,:3].T).T
            pts = apply_noise_to_object(pts, rng)

            original_point_count = len(pts); 
            
            h, w, l=map(float, label.split()[8:11]); 
            z_min_donor = pts[:, 2].min()
            x, y, yaw = sample_pose_by_class(cls, rng, sampler_pool)

            if x is None: 
                continue
            
            # Dynamic range check
            point_count = len(pts)
            if point_count > 150: 
                preferred_range=(0,15)
            elif point_count > 80: 
                preferred_range=(10,20)
            else: 
                preferred_range=(20,35)

            if not(preferred_range[0]<np.linalg.norm([x,y])<preferred_range[1]): 
                continue
            
            a, b, c, d = global_plane_params

            if abs(c)<1e-6: 
                continue

            global_ground_z= -(a * x + b * y + d)/ c
            z = global_ground_z + (h/2)
            box = Box(x, y, z, l, w, h, yaw+original_yaw_velo)

            # validation checks
            if not is_placement_realistic(box,cls,original_pc): 
                continue
            if any(boxes_overlap(box,b) for b in scene_boxes): 
                continue
            if len(get_points_in_box(original_pc,box))>10: 
                continue
            
            # Transform, then check partial occlusion
            Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]], np.float32)

            # Adjust points based on the label offsets and global ground plane
            # Convert the (x, y) offsets from camera to lidar frame using inverse transform

            pts[:,2] -= z_min_donor
            pts[:,:3] = (Rz@pts[:,:3].T).T
 
            pts[:,:3] += np.array([x, y, global_ground_z])
            # depth_map = build_depth_map(original_pc)
            # pts = is_line_of_sight_clear_spherical(original_pc, pts[:,:3], depth_map)
            # Add semantic channels and return the finished object
            if pts.shape[1] == 3: 
                pts = np.hstack([pts, 0.5 * np.ones((pts.shape[0], 1), dtype=np.float32)])
            sem_pts = np.zeros((pts.shape[0], 5), np.float32)
            sem_idx = {"Car": 1,"Pedestrian": 2,"Cyclist": 3}.get(cls, 4)
            sem_pts[:,sem_idx] = 1.0
            pts = np.hstack([pts, sem_pts])
            
            return {'pts': pts, 'label': label, 'box': box}

        return None