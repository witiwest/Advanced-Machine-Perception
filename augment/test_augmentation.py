from pathlib import Path
from collections import namedtuple
import numpy as np
import random
import pickle
import os
import glob

from validation_visualizer import create_validation_plot, split_cloud

# Some random frames
FRAMES     = ["00100", "00242", "00376", "00550", "00816", "01100", "01450", "01900", "02300"]
OBJ_CLASS  = "Pedestrian"                     # Class to paste: "Car", "Pedestrian", "Cyclist"
MAX_TRIAL  = 50                               # Attempts per frame
MARGIN_XY  = 0.15                             # SAT buffer

_HOME      = Path.home()
DATA_ROOT  = _HOME / "final_assignment" / "data" / "view_of_delft"
OBJ_DICT   = _HOME / "final_assignment" / "object_dict.pkl"
AUG_DIR    = _HOME / "final_assignment" / "augmented_frames"
FIG_DIR    = _HOME / "final_assignment" / "tests" / "figures"
AUG_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# geometry helpers 
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

def boxes_overlap(a: Box, b: Box, m_xy=MARGIN_XY, m_z=0.1):
    """ Checks for 3D overlap between two boxes using BEV SAT + Z-axis check. """
    z_ok = abs(a.z - b.z) < (a.h + b.h) / 2 + m_z
    return z_ok and _overlap_bev(a, b, m_xy)

# realistic-placement helpers
def is_line_of_sight_clear(pc: np.ndarray, point: np.ndarray, margin=0.5):
    """
    Checks if the line of sight from origin (0,0,0) to a target point
    is clear of other points in the point cloud.
    """
    direction = point / (np.linalg.norm(point) + 1e-6)
    # Project all scene points onto the ray direction vector
    projections = pc[:,:3] @ direction
    
    # Get the distance of the target point along the ray
    point_dist = np.linalg.norm(point)
    
    # Select scene points that are "in front" of the target point
    relevant_points = pc[(projections > 0) & (projections < point_dist)]
    if len(relevant_points) == 0:
        return True # Nothing is in front
        
    # For those points, find their perpendicular distance to the ray
    # If any point is close to the ray, the LoS is blocked
    perp_distances = np.linalg.norm(relevant_points[:,:3] - 
                                   (relevant_points[:,:3] @ direction)[:, np.newaxis] * direction, axis=1)

    return not np.any(perp_distances < margin)

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
        if -5 < x < 5:
            return None, None, None 

        # Cars should have an orientation aligned with the road
        yaw = rng.normal(loc=0.0, scale=np.deg2rad(5))
        
        # Small random offset to simulate not being perfectly centered on a point
        x += rng.uniform(-0.5, 0.5)
        y += rng.uniform(-0.5, 0.5)

    elif cls == "Pedestrian" or cls == "Cyclist":
        # Pedestrians and cyclists can be closer, but not right on top of the car
        if -2 < x < 2:
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
    if len(ground_points) < 5: return False
    
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

# insertion
def insert_object(pc, labels, obj_db, cls, scene_boxes, calib_dict, rng, global_plane_params, sampler_pool, *, max_trials=MAX_TRIAL):
    """
    Selects a donor object and attempts to place it onto the pre-computed global ground plane.
    """
    # This check is crucial: if no global plane was found, we can't do anything.
    if global_plane_params is None:
        return pc, labels, scene_boxes, False

    donors = obj_db.get(cls, [])
    if not donors:
        return pc, labels, scene_boxes, False

    # Main loop to try multiple random poses
    for _ in range(max_trials):
        
        # Random object selection for variety 
        ent = rng.choice(donors)
        pts = ent["points"].copy()

        # Apply noise for robustness
        pts = apply_noise_to_object(pts, rng)

        if len(pts) == 0: 
            continue # All points were dropped, try again

        label = ent["label"]
        h, w, l = map(float, label.split()[8:11])
        z_min_donor = pts[:, 2].min()

        x, y, yaw = sample_pose_by_class(cls, rng, sampler_pool)

        # if sampler rejected the pose
        if x is None:
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
        if not is_placement_realistic(box, cls, pc):                    
            continue
        
        if any(boxes_overlap(box, b) for b in scene_boxes):             
            continue
        
        points_in_box_indices = get_points_in_box(pc, box)
        if len(points_in_box_indices) > 5:                              
            continue
        
        if not is_line_of_sight_clear(pc, np.array([box.x, y, z])):      
            continue

        # Hole cutting an paste object
        pc = np.delete(pc, points_in_box_indices, axis=0)

        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw),  np.cos(yaw), 0], [0, 0, 1]], np.float32)
        centroid_xy = np.mean(pts[:, :2], axis=0)
        pts[:, 0] -= centroid_xy[0]
        pts[:, 1] -= centroid_xy[1]
        pts[:, 2] -= z_min_donor
        pts[:, :3] = (Rz @ pts[:, :3].T).T
        pts[:, :3] += np.array([x, y, global_ground_z])

        # Finalize point cloud and labels
        if pts.shape[1] == 3: 
            pts = np.hstack([pts, 0.5 * np.ones((pts.shape[0], 1), dtype=np.float32)])

        sem = np.zeros((pts.shape[0], 5), np.float32)
        sem_idx = {"Car": 1, "Pedestrian": 2, "Cyclist": 3}.get(cls, 4)
        sem[:, sem_idx] = 1.0

        if pc.shape[1] == 4:
            base_sem = np.zeros((pc.shape[0], 5), np.float32) 
            base_sem[:, 0] = 1.0
            pc = np.hstack([pc, base_sem])

        pts = np.hstack([pts, sem])
        pc = np.vstack([pc, pts])
        labels.append(label)
        scene_boxes.append(box)

        # Valid pose was found
        return pc, labels, scene_boxes, True 
    
    # Max trials reached, no success
    return pc, labels, scene_boxes, False

# main loop 
if __name__ == "__main__":
    print("Loading object database...")
    with open(OBJ_DICT, "rb") as f:
        OBJ_DB = pickle.load(f)
    print(f"Loaded {sum(len(v) for v in OBJ_DB.values())} objects across {len(OBJ_DB)} classes.")

    # For fixed debugging
    rng = np.random.default_rng(42)

    for frame in FRAMES:
        print(f"Augmenting frame {frame} with a '{OBJ_CLASS}'...")
        lidar_path = DATA_ROOT / "lidar/training/velodyne" / f"{frame}.bin"
        label_path = DATA_ROOT / "lidar/training/label_2" / f"{frame}.txt"
        calib_path = DATA_ROOT / "lidar/training/calib" / f"{frame}.txt"
        rgb_path   = DATA_ROOT / "lidar/training/image_2" / f"{frame}.jpg"

        pc   = load_point_cloud(lidar_path)
        lab  = load_labels(label_path)
        calib_dict = load_kitti_calib(calib_path)

        # Calculate the single global ground plane for the entire scene.
        global_plane_params = fit_global_ground_plane_ransac(pc)

        sampler_pool = get_data_driven_sampler_pool(pc)

        Tr_cam_to_velo = np.linalg.inv(np.vstack([calib_dict["Tr_velo_to_cam"], [0, 0, 0, 1]]))
        
        scene_boxes = []
        for ln in lab:
            p = ln.split()
            h_c, w_c, l_c = map(float, p[8:11]); x_c, y_c, z_c = map(float, p[11:14]); ry_c = float(p[14])
            loc_velo = (Tr_cam_to_velo @ np.array([x_c, y_c, z_c, 1.0]))[:3]
            scene_boxes.append(Box(loc_velo[0], loc_velo[1], loc_velo[2], l_c, w_c, h_c, -(ry_c + np.pi/2)))

        # Pass the computed global_plane_params to the insertion function.
        pc_aug, lab_aug, _, ok = insert_object(
            pc.copy(), lab.copy(), OBJ_DB, OBJ_CLASS, scene_boxes, calib_dict, rng, global_plane_params, sampler_pool)

        if ok:
            aug_bin_path = AUG_DIR / f"{frame}_aug.bin"
            save_point_cloud(aug_bin_path, pc_aug)
            save_labels(AUG_DIR / f"{frame}_aug.txt", lab_aug)
            print(f"   - Inserted & saved to {aug_bin_path.name}")
            
            orig_xyz, ins_xyz = split_cloud(pc_aug)
            val_plot_path = FIG_DIR / f"{frame}_{OBJ_CLASS.lower()}_validation.png"
            create_validation_plot(
                original_xyz=orig_xyz,
                inserted_xyz=ins_xyz,
                image_path=rgb_path,
                calib=calib_dict,
                save_path=val_plot_path,
                obj_class=OBJ_CLASS
            )
        else:
            print("No valid pose found after all trials.")

    print("\nDone.")