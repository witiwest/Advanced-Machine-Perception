from pathlib import Path
from collections import namedtuple
import numpy as np
import random
import pickle
import os
import glob
from scipy.spatial import KDTree

Box = namedtuple("Box", "x y z l w h yaw")

def fit_global_ground_plane_ransac(pc: np.ndarray, iters=100, eps=0.1):
    """
    Fits a single, global ground plane to the entire scene using RANSAC.

    It focuses on points near the ego-vehicle to get a stable estimate, assuming
    this represents the primary ground plane.
    """
    # First, filter the point cloud to a "trusted" subset for finding the ground.
    # We use points that are relatively close to the vehicle and below the sensor height.
    trusted_mask = (np.linalg.norm(pc[:, :2], axis=1) < 20.0) & (pc[:, 2] < -0.5)
    ground_candidates = pc[trusted_mask, :3]

    if len(ground_candidates) < 50:
        print("Warning: Not enough ground points near vehicle to fit a global plane.")
        return None

    # RANSAC Implementation 
    best_inliers_count = 0
    best_plane = None
    rng = np.random.default_rng()

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

def is_line_of_sight_clear(pc: np.ndarray, object: np.ndarray, margin=0.05, num_ray_samples=200):
    pc_xyz = pc[:, :3] if pc.shape[1] > 3 else pc
    voxel_set = build_voxel_hash(pc_xyz, margin)

    visible, occluded = [], []

    for point in object:
        norm = np.linalg.norm(point)
        if norm < 1e-8:
            visible.append(point)
            continue

        ray_samples = point * np.linspace(0.1, 0.9, num_ray_samples)[:, None]
        ray_voxels = [get_voxel_key(s, margin) for s in ray_samples]

        if any(v in voxel_set for v in ray_voxels):
            occluded.append(point)
        else:
            visible.append(point)

    print(f"{len(visible)} points are clear, {len(occluded)} are occluded.")
    return np.array(visible), np.array(occluded)

def get_data_driven_sampler_pool(pc: np.ndarray):
    """
    Filters the scene's point cloud to find all points that are likely on a
    traversable ground surface, creating a pool for data-driven sampling.
    """
    # Exclude points that are too close (ego-vehicle), too high (buildings),
    # or potentially part of the sky.
    ground_mask = (pc[:, 2] < -0.5) & (np.linalg.norm(pc[:, :2], axis=1) > 10)
    return pc[ground_mask, :2]

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
        # Enforce a "dead zone" immediately around the ego-vehicle for cars.
        if x < 5 or x > 30 or y > 0.5 * x or y < -0.5 * x:
            return None, None, None 

        # Cars should have an orientation aligned with the road
        yaw = rng.normal(loc=0.0, scale=np.deg2rad(5))
        
        # Small random offset to simulate not being perfectly centered on a point
        x += rng.uniform(-0.5, 0.5)
        y += rng.uniform(-0.5, 0.5)

    elif cls == "Pedestrian" or cls == "Cyclist":
        # Pedestrians and cyclists can be closer, but not right on top of the car
        if x < 2 or x > 30 or y > 0.5 * x or y < -0.5 * x:
            return None, None, None # Reject if too close

        # Pedestrians can face any way, cyclists are mostly forward
        yaw = rng.uniform(-np.pi, np.pi) if cls == "Pedestrian" else rng.normal(loc=0.0, scale=np.deg2rad(20))

    return x, y, yaw

def is_placement_realistic(box: Box, cls: str, scene_pc: np.ndarray):
    """ Context-aware placement check using local ground geometry. """
    local_patch_radius = 1.5
    distances = np.linalg.norm(scene_pc[:, :2] - np.array([box.x, box.y]), axis=1)
    local_points = scene_pc[distances < local_patch_radius]
    if len(local_points) < 10: return False

    ground_points = local_points[local_points[:, 2] < (local_points[:, 2].min() + 0.25)]
    if len(ground_points) < 10: return False
    
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

def remove_occluded_points(pc: np.ndarray, aabb_min: np.ndarray, aabb_max: np.ndarray, sensor_origin=np.array([0.0, 0.0, 0.0])):
    """
    Remove points occluded by the inserted object along LoS from sensor.
    
    Args:
        pc: (N,4) LiDAR points
        box_center: (3,) center of inserted object
        box_dims: (width, length, height) of object
        sensor_origin: sensor coordinates (default at origin)
    
    Returns:
        filtered_pc: points with occluded points removed
    """

    # Calculate corners of the box in XY plane (assuming axis aligned)
    x0, y0 = aabb_min
    x1, y1 = aabb_max
    corners = np.array([
        [x0, y0],
        [x0, y1],
        [x1, y1],
        [x1, y0],
    ], dtype=float)  # shape (4,2)
    
    # 2) angles & ranges of corners
    vecs = corners - sensor_origin[:2]    # (4,2)
    corner_angles = np.arctan2(vecs[:,1], vecs[:,0])  # (4,)
    corner_ranges = np.linalg.norm(vecs, axis=1)      # (4,)

    # normalize to [-pi,pi]
    def norm(a): return (a + np.pi) % (2*np.pi) - np.pi
    corner_angles = norm(corner_angles)
    min_ang, max_ang = corner_angles.min(), corner_angles.max()
    near_range = corner_ranges.min()  # distance to the nearest bbox corner

    # 3) scene pts in polar
    pts_xy = pc[:, :2] - sensor_origin[:2]
    pt_angles = norm(np.arctan2(pts_xy[:,1], pts_xy[:,0]))
    pt_ranges = np.linalg.norm(pts_xy, axis=1)

    # 4) build the angular mask (handles wrap-around)
    if min_ang < max_ang:
        in_sector = (pt_angles >= min_ang) & (pt_angles <= max_ang)
    else:
        in_sector = (pt_angles >= min_ang) | (pt_angles <= max_ang)

    # 5) anything in that wedge and farther than the near corner is occluded
    occluded = in_sector & (pt_ranges > near_range + 1e-6)

    # 6) filter out occluded
    return pc[~occluded]

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
        """
        The main method called by the data loader for each sample.
        """
        # Randomly decide whether to augment this scene at all
        if self.rng.random() > self.augmentation_prob:
            return data_sample

        # Prepare data for augmentation
        pc = data_sample['pc'].copy()
        labels = data_sample['labels'].copy()
        calib = data_sample['calib']
        cls_to_insert = self.rng.choice(self.classes_to_augment)

        # Perform pre-computations needed for insertion
        global_plane_params = fit_global_ground_plane_ransac(pc)
        sampler_pool = get_data_driven_sampler_pool(pc)
        Tr_cam_to_velo = np.linalg.inv(np.vstack([calib["Tr_velo_to_cam"], [0, 0, 0, 1]]))
        scene_boxes = []
        for ln in labels:
            p = ln.split()
            if p[0] not in ['Car', 'Pedestrian', 'Cyclist']: 
                continue
            h_c, w_c, l_c = map(float, p[8:11]); x_c, y_c, z_c = map(float, p[11:14]); ry_c = float(p[14])
            loc_velo = (Tr_cam_to_velo @ np.array([x_c, y_c, z_c, 1.0]))[:3]
            scene_boxes.append(Box(loc_velo[0], loc_velo[1], loc_velo[2], l_c, w_c, h_c, -(ry_c + np.pi/2)))

        # Multi-object insertion loop
        num_placed = 0
        # print(f"Attempting Augmentation for Frame")
        if self.cfg.multi_object.enabled:
            max_objects = self.cfg.multi_object.max_objects
            for i in range(max_objects):
                # Check probability of attempting this insertion
                # print(f"  [Attempt {i+1}/{max_objects}]")
                if self.rng.random() > self.cfg.multi_object.attempt_probs[i]:
                    break 
                
                # print(f"Probabilistic check passed. Trying to insert object #{i+1}...")
                cls_to_insert = self.rng.choice(self.classes_to_augment)
                
                # Call our existing insertion logic. Note that we pass the current state
                # of pc, labels, and scene_boxes to it.
                pc_new, labels_new, scene_boxes_new, ok = self._perform_insertion(
                    pc, labels, self.obj_db, cls_to_insert, scene_boxes, calib, self.rng, global_plane_params, sampler_pool)

                if ok:
                    # print(f"SUCCESS: Placed a '{cls_to_insert}'. Updating scene state for next attempt.")
                    # Update the state for the next iteration
                    pc = pc_new
                    labels = labels_new
                    scene_boxes = scene_boxes_new
                    num_placed += 1
                else:
                    # print(f"FAILURE: No valid pose found after {self.max_trials} trials. Scene may be too crowded. Stopping.")
                    # If one attempt fails, we assume the scene is too crowded and stop.
                    break
        # print(f"Finished Augmentation. Total objects placed: {num_placed}")
        # If at least one object was successfully placed, update the final data sample
        if num_placed > 0:
            data_sample['pc'] = pc
            data_sample['labels'] = labels
    
        return data_sample
    
    def _perform_insertion(self, pc, labels, obj_db, cls, scene_boxes, calib_dict, rng, global_plane_params, sampler_pool):
        """
        Selects a donor object and attempts to place it onto the pre-computed global ground plane.
        """
        # This check is crucial: if no global plane was found, we can't do anything.
        if global_plane_params is None:
            return pc, labels, scene_boxes, False

        donors = obj_db.get(cls, [])
        if not donors:
            return pc, labels, scene_boxes, False
        
        original_scene_pc = pc.copy()

        # Main loop to try multiple random poses
        for _ in range(self.max_trials):
            
            # Random object selection for variety 
            ent = rng.choice(donors)
            pts = ent["points"].copy()

            # Apply noise for robustness
            pts = apply_noise_to_object(pts, rng)
            point_count = len(pts)
            if point_count > 150:
                preferred_range = (2, 15)
            elif point_count > 80:
                preferred_range = (15, 25)
            else:
                preferred_range = (25, 35)
            # print(f"Trying to insert {cls} with {point_count} points...")


            if len(pts) == 0: 
                continue # All points were dropped, try again

            label = ent["label"]
            h, w, l = map(float, label.split()[8:11])
            z_min_donor = pts[:, 2].min()

            x, y, yaw = sample_pose_by_class(cls, rng, sampler_pool)

            # if sampler rejected the pose
            if x is None:
                continue

            if not (preferred_range[0] < np.linalg.norm([x, y]) < preferred_range[1]):
                continue

            # Calculate the Z-height directly from the global plane equation
            a, b, c, d = global_plane_params
            # Check for a near-zero 'c' to avoid division by zero if the plane is vertical
            if abs(c) < 1e-6:
                continue

            global_ground_z = -(a * x + b * y + d) / c

            # Final object position
            z = global_ground_z + (h / 2) - z_min_donor
            box = Box(x, y, z, l, w, h, yaw)

            # Validation checks
            if not is_placement_realistic(box, cls, original_scene_pc):                    
                continue
            
            if any(boxes_overlap(box, b) for b in scene_boxes):             
                continue
            
            points_in_box_indices = get_points_in_box(original_scene_pc, box)
            if len(points_in_box_indices) > 5:                              
                continue

            Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw),  np.cos(yaw), 0], [0, 0, 1]], np.float32)
            centroid_xy = np.mean(pts[:, :2], axis=0)
            pts[:, 0] -= centroid_xy[0]
            pts[:, 1] -= centroid_xy[1]
            pts[:, 2] -= z_min_donor
            pts[:, :3] = (Rz @ pts[:, :3].T).T
            pts[:, :3] += np.array([x, y, global_ground_z])

            # print(f"len(pts) before:{len(pts)}")
            pts, occluded_points = is_line_of_sight_clear(original_scene_pc, pts, margin=0.05)
            # print(f"len(pts) after:{len(pts)}")
            if len(occluded_points) > len(pts):
                continue
            if len(pts) == 0: 
                continue
            pc_after_occlusion = remove_occluded_points(original_scene_pc, pts[:, :2].min(0), pts[:, :2].max(0))

            # Finalize point cloud and labels
            if pts.shape[1] == 3: 
                pts = np.hstack([pts, 0.5 * np.ones((pts.shape[0], 1), dtype=np.float32)])

            sem = np.zeros((pts.shape[0], 5), np.float32)
            sem_idx = {"Car": 1, "Pedestrian": 2, "Cyclist": 3}.get(cls, 4)
            sem[:, sem_idx] = 1.0

            if pc_after_occlusion.shape[1] == 4:
                base_sem = np.zeros((pc_after_occlusion.shape[0], 5), np.float32) 
                base_sem[:, 0] = 1.0
                pc_final = np.hstack([pc_after_occlusion, base_sem])
            else:
                pc_final = pc_after_occlusion

            pts = np.hstack([pts, sem])
            # print(f"len(pc), len(pts) before:{len(pc_final)}, {len(pts)}")
            pc_final = np.vstack([pc_final, pts])
            # print(f"len(pc):{len(pc_final)}")
            labels.append(label)
            scene_boxes.append(box)

            # Valid pose was found
            return pc_final, labels, scene_boxes, True 
        
        # Max trials reached, no success
        return pc, labels, scene_boxes, False