import numpy as np
import pickle
import random

def load_point_cloud(bin_path):
    return np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)

def save_point_cloud(bin_path, points):
    points.astype(np.float32).tofile(bin_path)

def load_labels(label_path):
    with open(label_path, 'r') as f:
        return f.readlines()

def save_labels(label_path, labels):
    with open(label_path, 'w') as f:
        f.writelines(labels)

def insert_object_into_point_cloud(point_cloud, label_lines, object_dict_path, obj_class):
    # Load object dict
    with open(object_dict_path, 'rb') as f:
        object_dict = pickle.load(f)

    # Select random object of the given class
    obj_list = object_dict.get(obj_class, [])

    if not obj_list:
        raise ValueError(f"No objects found for class '{obj_class}' in object_dict.")
    
    obj = random.choice(obj_list)
    print(obj)
    
    obj_points = obj['points']  # shape (N, 3)
    obj_label = obj['label']    # string in KITTI format
    # print(obj)
    # Add reflectance column if needed
    if obj_points.shape[1] == 3:
        reflectance = np.ones((obj_points.shape[0], 1)) * 0.5
        obj_points = np.hstack([obj_points, reflectance])

    # Merge point cloud
    new_point_cloud = np.vstack([point_cloud, obj_points])
    
    # Append label
    label_lines.append(obj_label if obj_label.endswith('\n') else obj_label + '\n')
    
    return new_point_cloud, label_lines


# Example usage:
lidar_path = '/home/taalbers/final_assignment/data/view_of_delft/lidar/training/velodyne/00000.bin'
label_path = '/home/taalbers/final_assignment/data/view_of_delft/lidar/training/label_2/00000.txt'
object_dict_path = '/home/taalbers/final_assignment/object_dict.pkl'

point_cloud = load_point_cloud(lidar_path)
label_lines = load_labels(label_path)

new_pc, new_labels = insert_object_into_point_cloud(point_cloud, label_lines, object_dict_path, 'Cyclist')

# Save new files
save_point_cloud('/home/taalbers/00000_augmented.bin', new_pc)
save_labels('/home/taalbers/00000_augmented.txt', new_labels)

import matplotlib.pyplot as plt


def visualize_point_cloud(points, title):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.5, c='blue')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def visualize_augmented_point_cloud(all_points, original_points_count, title):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    # Original points in blue, small
    ax.scatter(all_points[:original_points_count, 0], all_points[:original_points_count, 1], all_points[:original_points_count, 2], 
               s=0.5, c='blue', label='Original')
    # Added points in red, larger
    ax.scatter(all_points[original_points_count:, 0], all_points[original_points_count:, 1], all_points[original_points_count:, 2], 
               s=8, c='red', label='Inserted Object')
    ax.invert_yaxis()

    # Set viewing angle: elev (up/down), azim (left/right)
    ax.view_init(elev=0, azim=-90)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

# Visualize original and augmented point clouds
visualize_point_cloud(point_cloud, "Original Point Cloud")
visualize_augmented_point_cloud(new_pc, point_cloud.shape[0], "Augmented Point Cloud (Object Points in Red)")

# Print label data before and after
# print("Original Labels:")
# print(''.join(label_lines))
# print("\nAugmented Labels:")
# print(''.join(new_labels))