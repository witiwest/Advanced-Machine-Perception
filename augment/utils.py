import numpy as np

def apply_global_transformations(points, gt_boxes, rotation_angle, scaling_factor):
    """
    Applies a global rotation and scaling to both the point cloud and the ground truth boxes.
    """
    # Create the rotation matrix for the Z-axis
    c, s = np.cos(rotation_angle), np.sin(rotation_angle)
    rotation_matrix = np.array([[c, -s, 0], [s,  c, 0], [0,  0, 1]], dtype=np.float32)
    
    # 1. Apply transformations to the point cloud
    final_points = points.copy()
    final_points[:, :3] *= scaling_factor
    final_points[:, :3] = final_points[:, :3] @ rotation_matrix.T

    # 2. Apply transformations to the ground truth boxes
    if gt_boxes.shape[0] > 0:
        final_gt_boxes = gt_boxes.copy()
        final_gt_boxes[:, :6] *= scaling_factor
        final_gt_boxes[:, :3] = final_gt_boxes[:, :3] @ rotation_matrix.T
        final_gt_boxes[:, 6] += rotation_angle
    else:
        final_gt_boxes = gt_boxes

    return final_points, final_gt_boxes

