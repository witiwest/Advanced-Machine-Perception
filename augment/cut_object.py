# IMPORT MODULES
import numpy as np
# from scipy.spatial.transform import Rotation as R
import struct
import os
import glob
import random
import pickle

# ---------------------------------------------------------------------
"""
Input: Point cloud data from KITTI dataset, annotations in text files, and calibration matrices.
Output: Extracted objects from point clouds based on annotations, saved in a dictionary format.
This script reads point clouds and annotations from the KITTI dataset, extracts objects based on the annotations, and saves them in a dictionary format.
"""
# ---------------------------------------------------------------------


def load_calib_Tr_velo_to_cam(calib_filepath):
    """
    Reads the KITTI calib.txt, finds the line starting with 'Tr_velo_to_cam:', 
    parses the 12 floats into a 3×4, then returns a 4×4 by appending [0 0 0 1].
    """
    Tr = None
    with open(calib_filepath, 'r') as f:
        for line in f:
            if line.startswith('Tr_velo_to_cam:'):
                parts = line.strip().split()[1:]  # skip the 'Tr_velo_to_cam:' token
                vals = np.array([float(x) for x in parts]).reshape(3, 4)
                Tr = np.vstack([vals, np.array([0.0, 0.0, 0.0, 1.0])])  # now 4×4
                break
    if Tr is None:
        raise RuntimeError(f"Could not find 'Tr_velo_to_cam' in {calib_filepath}")
    return Tr

def invert_Tr_velo_to_cam(Tr_velo_to_cam):
    """
    Given 4×4 = [R | t; 0 0 0 1], return its inverse (cam → velo).
    """
    return np.linalg.inv(Tr_velo_to_cam)

def parse_one_kitti_line(line):
    """
    Input: one whitespace‐separated string of length ≥15.
    Returns a dict of raw camera‐frame values.
    """
    parts = line.strip().split()

    raw_ann = {
        'type':      parts[0],                 
        'truncated': float(parts[1]),          
        'occluded':  int(parts[2]),
        'alpha':     float(parts[3]),
        'bbox': {
            'xmin': float(parts[4]),
            'ymin': float(parts[5]),
            'xmax': float(parts[6]),
            'ymax': float(parts[7]),
        },
        'dimensions_cam': {
            'h': float(parts[8]),   
            'w': float(parts[9]),   
            'l': float(parts[10])   
        },
        'location_cam': {
            'x': float(parts[11]),  
            'y': float(parts[12]),  
            'z': float(parts[13])   
        },
        'rotation_y': float(parts[14])  # yaw around Y axis in camera coords
    }
    return raw_ann

def cam_to_velo(pt_cam, Tr_cam_to_velo):
    """
    pt_cam: numpy array [x_cam, y_cam, z_cam].
    Tr_cam_to_velo: 4×4 matrix.
    Returns [x_velo, y_velo, z_velo].
    """
    xyz1 = np.array([pt_cam[0], pt_cam[1], pt_cam[2], 1.0])  # homogeneous
    velo_hom = Tr_cam_to_velo @ xyz1                         
    return velo_hom[:3] / velo_hom[3] 

def cam_yaw_to_velo_quaternion(rotation_y_cam):
    """
    rotation_y_cam: float (radians), yaw around camera's Y.
    Returns: quaternion [qx, qy, qz, qw] for rotating in velodyne frame.
    """
    yaw_velo = -(rotation_y_cam + np.pi/2)
    return euler_z_to_quaternion(yaw_velo) 

# Math helper functions
def euler_z_to_quaternion(yaw):
    # Only for rotation around Z axis (as in your case)
    qw = np.cos(yaw / 2)
    qz = np.sin(yaw / 2)
    return np.array([0.0, 0.0, qz, qw])

def quaternion_to_rot_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
    ])

def build_velo_annotation(raw_ann, Tr_cam_to_velo):
    """
    raw_ann: dict from parse_one_kitti_line(...)
    Tr_cam_to_velo: 4×4 cam→velo homogeneous matrix
    
    Returns a dict with keys:
      'center':   {'x': x_velo, 'y': y_velo, 'z': z_velo},
      'rotation': {'x': Qx,    'y': Qy,    'z': Qz,    'w': Qw},
      'length': <float>,  # KITTI length
      'width':  <float>,  # KITTI width
      'height': <float>   # KITTI height
    """
    # CONVERT LOCATION FROM CAMERA FRAME TO VELODYNE FRAME
    loc_cam = np.array([
        raw_ann['location_cam']['x'],
        raw_ann['location_cam']['y'],
        raw_ann['location_cam']['z']
    ])
    x_velo, y_velo, z_velo = cam_to_velo(loc_cam, Tr_cam_to_velo)
    
    # CONVERT ROTATION FROM CAMERA FRAME TO VELODYNE FRAME
    qx, qy, qz, qw = cam_yaw_to_velo_quaternion(raw_ann['rotation_y'])
    
    # REORDER DIMENSIONS FROM CAMERA FRAME TO VELODYNE FRAME
    length_velo = raw_ann['dimensions_cam']['l']  # KITTI length
    width_velo  = raw_ann['dimensions_cam']['w']  # KITTI width
    height_velo = raw_ann['dimensions_cam']['h']  # KITTI height
    
    return {
        'center': {
            'x': float(x_velo),
            'y': float(y_velo),
            'z': float(z_velo)
        },
        'rotation': {
            'x': float(qx),
            'y': float(qy),
            'z': float(qz),
            'w': float(qw)
        },
        'length': float(length_velo),
        'width':  float(width_velo),
        'height': float(height_velo)
    }

# This function was taken from: https://github.com/ctu-vras/pcl-augmentation/blob/main/object_detection/Real3DAug/tools/cut_bbox.py
def cut_bounding_box(point_cloud, annotation, annotation_move=[0, 0, 0]):
    """
    Function, which cuts bounding box from point-cloud.
    :param point_cloud: numpy 2D array, original point-cloud
    :param annotation: dictionary, annotation of bounding box which should be cut out
    :param annotation_move: numpy 1D array, row translation vector between annotation and LiDAR
    :return: numpy 2D array, annotations point-cloud
    """
    xc = annotation['center']['x'] - annotation_move[0]
    yc = annotation['center']['y'] - annotation_move[1]
    zc = annotation['center']['z'] - annotation_move[2]
    Q1 = annotation['rotation']['x']
    Q2 = annotation['rotation']['y']
    Q3 = annotation['rotation']['z']
    Q4 = annotation['rotation']['w']
    length = annotation['length']
    width = annotation['width']
    height = annotation['height']

    q = [annotation['rotation']['x'], annotation['rotation']['y'], annotation['rotation']['z'], annotation['rotation']['w']]
    rot_matrix = quaternion_to_rot_matrix(q)

    bbox = point_cloud[
        rot_matrix[0][0] * point_cloud[:, 0] + rot_matrix[1][0] * point_cloud[:, 1] + rot_matrix[2][
            0] * point_cloud[:,
                 2] <
        rot_matrix[0][0] * (xc + rot_matrix[0][0] * length / 2) + rot_matrix[1][0] * (
                yc + rot_matrix[1][0] * length / 2) + rot_matrix[2][0] *
        (zc + rot_matrix[2][0] * length / 2)]

    bbox = bbox[
        rot_matrix[0][0] * bbox[:, 0] + rot_matrix[1][0] * bbox[:, 1] + rot_matrix[2][0] * bbox[:, 2] >
        rot_matrix[0][0] * (xc - rot_matrix[0][0] * length / 2) + rot_matrix[1][0] * (
                yc - rot_matrix[1][0] * length / 2) + rot_matrix[2][0] *
        (zc - rot_matrix[2][0] * length / 2)]

    bbox = bbox[
        rot_matrix[0][1] * bbox[:, 0] + rot_matrix[1][1] * bbox[:, 1] + rot_matrix[2][1] * bbox[:, 2] <
        rot_matrix[0][1] * (xc + rot_matrix[0][1] * width / 2) + rot_matrix[1][1] * (
                yc + rot_matrix[1][1] * width / 2) + rot_matrix[2][1] *
        (zc + rot_matrix[2][1] * width / 2)]

    bbox = bbox[
        rot_matrix[0][1] * bbox[:, 0] + rot_matrix[1][1] * bbox[:, 1] + rot_matrix[2][1] * bbox[:, 2] >
        rot_matrix[0][1] * (xc - rot_matrix[0][1] * width / 2) + rot_matrix[1][1] * (
                yc - rot_matrix[1][1] * width / 2) + rot_matrix[2][1] *
        (zc - rot_matrix[2][1] * width / 2)]

    bbox = bbox[
        rot_matrix[0][2] * bbox[:, 0] + rot_matrix[1][2] * bbox[:, 1] + rot_matrix[2][2] * bbox[:, 2] <
        rot_matrix[0][2] * (xc + rot_matrix[0][2] * height) + rot_matrix[1][2] * (
                yc + rot_matrix[1][2] * height) + rot_matrix[2][2] *
        (zc + rot_matrix[2][2] * height)]

    bbox = bbox[
        rot_matrix[0][2] * bbox[:, 0] + rot_matrix[1][2] * bbox[:, 1] + rot_matrix[2][2] * bbox[:, 2] >
        rot_matrix[0][2] * (xc - rot_matrix[0][2] * 0) + rot_matrix[1][2] * (
                yc - rot_matrix[1][2] * 0) + rot_matrix[2][2] *
        (zc - rot_matrix[2][2] * 0)]

    return bbox


def create_cut_out_object_dictionary(training_data, annotations, annotation_move=[0, 0, 0]):
    """
    Loops over training data and extracts all objects using cut_bounding_box.
    :param training_data: list of numpy arrays, each is a point cloud
    :param annotations: list of lists of annotation dicts, each sublist corresponds to objects in a point cloud
    :param annotation_move: translation vector
    :return: dict mapping (cloud_idx, obj_idx) to extracted object point clouds
    """
    object_dict = {}
    for cloud_idx, (point_cloud, cloud_annotations) in enumerate(zip(training_data, annotations)):
        for obj_idx, annotation in enumerate(cloud_annotations):
            obj_points = cut_bounding_box(point_cloud, annotation, annotation_move)
            object_dict[(cloud_idx, obj_idx)] = obj_points
    return object_dict

def run(training_data, annotations, annotation_move=[0, 0, 0]):
    """
    Main function to run the cut object extraction.
    :param training_data: list of numpy arrays, each is a point cloud
    :param annotations: list of lists of annotation dicts, each sublist corresponds to objects in a point cloud
    :param annotation_move: translation vector
    :return: dict mapping (cloud_idx, obj_idx) to extracted object point clouds
    """
    return create_cut_out_object_dictionary(training_data, annotations, annotation_move)


# DATA DIRECTORIES
_HOME = os.path.expanduser('~')
_BASE = os.path.join(_HOME, 'final_assignment', 'view_of_delft', 'lidar', 'training')

label_data_dir = os.path.join(_BASE, 'label_2')
lidar_data_dir = os.path.join(_BASE, 'velodyne')
calib_data_dir = os.path.join(_BASE, 'calib')

# SELECT 100 RANDOM POINT CLOUDS TO EXTRACT OBJECTS FROM
all_txt = sorted(glob.glob(os.path.join(label_data_dir, '*.txt')))
random_txt = random.sample(all_txt, 500) 

random_bin   = []
random_calib = []
for txt_path in random_txt:
    base = os.path.splitext(os.path.basename(txt_path))[0]
    random_bin.append(os.path.join(lidar_data_dir, base + '.bin'))
    random_calib.append(os.path.join(calib_data_dir, base + '.txt'))

# LOAD SELECTED POINT CLOUDS FROM BINARY FILES
def load_one_bin(path):
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]
    return pts

training_data = [load_one_bin(p) for p in random_bin]

# CREATE ANNOTATIONS FOR EACH TRAINING FRAME
all_converted_annotations = []
for txt_path, calib_path in zip(random_txt, random_calib):
    # LOAD THE CALIBRATION MATRIX TR_velo_to_cam AND INVERT IT
    Tr_velo_to_cam = load_calib_Tr_velo_to_cam(calib_path)
    Tr_cam_to_velo = invert_Tr_velo_to_cam(Tr_velo_to_cam)
    
    # PARSE THE ANNOTATIONS FROM THE TXT FILE to raw_anns
    raw_anns = []
    with open(txt_path, 'r') as f:
        for line in f:
            if line.strip() == '':
                continue
            raw_anns.append(parse_one_kitti_line(line))
    
    # CONVERT raw_anns TO velo_anns USING THE CALIBRATION MATRIX
    velo_anns = [build_velo_annotation(r, Tr_cam_to_velo) for r in raw_anns]
    all_converted_annotations.append(velo_anns)

# OUTER LOOP OVER TRAINNING FRAMES AND INNER LOOP OVER ANNOTATIONS
INTEREST = {"Car", "Pedestrian", "Cyclist"}
object_dict = {}
for i, (pc, raw_anns, velo_ann_list) in enumerate(zip(training_data, [open(p).readlines() for p in random_txt], all_converted_annotations)):
    for j, (raw_line, velo_ann) in enumerate(zip(raw_anns, velo_ann_list)):
        pts_in_box = cut_bounding_box(pc, velo_ann, annotation_move=[0, 0, 0])
        parsed_class = raw_line.strip().split()[0]
        if parsed_class not in INTEREST:
            continue
        if len(pts_in_box) <= 8:
            continue  # Skip empty boxes
        
        entry = {
            'points': pts_in_box,
            'label': raw_line.strip()
        }

        if parsed_class not in object_dict:
            object_dict[parsed_class] = []

        object_dict[parsed_class].append(entry)
# PRINT NUMBER OF OBJECTS EXTRACTED
print(f"Extracted {len(object_dict)} objects from {len(training_data)} frames.")

# SAVE OBJECT DICTIONARIES TO PICKLE FILE ON DISK
print(object_dict["Cyclist"])
with open('object_dict.pkl', 'wb') as f:
    pickle.dump(object_dict, f)