from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Geometry and Projection Helpers

def split_cloud(augmented_points: np.ndarray):
    """
    Splits the augmented point cloud into original and inserted points
    using the semantic channels (assuming a 9-channel format).
    """
    # If cloud doesn't have semantic channels, it wasn't augmented.
    # Return the cloud as "original" and an empty array as "inserted".
    if augmented_points.shape[1] < 9:
        return augmented_points[:, :3], np.array([]).reshape(0, 3)

    # Original points are marked with a 1 in the 'unknown' semantic channel (column 4)
    original_mask = augmented_points[:, 4] == 1.0
    orig_xyz = augmented_points[original_mask, :3]

    # Inserted points are all other points (where column 4 is not 1)
    inserted_xyz = augmented_points[~original_mask, :3]

    return orig_xyz, inserted_xyz

def project_velo_to_image(xyz_velo: np.ndarray, calib: dict):
    """ Projects (N,3) LiDAR points to (N,2) pixel coordinates. """
    if xyz_velo.shape[0] == 0: return np.array([])
    N = xyz_velo.shape[0]
    
    tr_3x4 = calib["Tr_velo_to_cam"]
    tr_4x4 = np.vstack([tr_3x4, [0, 0, 0, 1]])
    
    xyz1 = np.hstack([xyz_velo, np.ones((N, 1), dtype=np.float32)])
    cam = (tr_4x4 @ xyz1.T)[:3, :]
    cam_rect = calib["R0_rect"] @ cam

    valid = cam_rect[2, :] > 0.1
    cam_rect[:, ~valid] = 0

    img_homo = calib["P2"] @ np.vstack([cam_rect, np.ones((1, N))])
    pix = (img_homo[:2] / img_homo[2, :]).T
    pix[~valid] = np.nan
    return pix

def make_axis_aligned_bbox(pts_velo: np.ndarray) -> np.ndarray:
    """ Builds an axis-aligned 3D box from a point cluster in the Velodyne frame. """
    if pts_velo.shape[0] == 0: return np.array([])
    xmin, ymin, zmin = pts_velo.min(0)
    xmax, ymax, zmax = pts_velo.max(0)
    return np.array([
        [xmax, ymax, zmin], [xmin, ymax, zmin], [xmin, ymin, zmin], [xmax, ymin, zmin],
        [xmax, ymax, zmax], [xmin, ymax, zmax], [xmin, ymin, zmax], [xmax, ymin, zmax],
    ], dtype=np.float32)

def _draw_3d_bbox(ax, pts2d, **kw):
    """ Draws the 12 lines of a 3D bbox in an image. """
    pairs = [
        (0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    for i, j in pairs:
        if not (np.isnan(pts2d[i]).any() or np.isnan(pts2d[j]).any()):
            ax.plot([pts2d[i, 0], pts2d[j, 0]], [pts2d[i, 1], pts2d[j, 1]], **kw)

# The Main Plotting Function

def create_validation_plot(
    original_xyz: np.ndarray,
    inserted_xyz: np.ndarray,
    image_path: Path,
    calib: dict,
    save_path: Path,
    obj_class
):
    """
    Generates a single validation image with two subplots:
    1. Left: Bird's-Eye-View (BEV) of the pasted object.
    2. Right: Camera overlay with the pasted object and its 3D bbox.
    """
    fig, (ax_bev, ax_cam) = plt.subplots(1, 2, figsize=(22, 7))
    plot_title = f"Validation for Frame: {image_path.stem} | Pasted: {obj_class}"
    fig.suptitle(plot_title, fontsize=16)

    # Plot 1: Bird's-Eye-View (BEV)
    ax_bev.scatter(original_xyz[:, 0], original_xyz[:, 1], s=1.5, c="royalblue", label="Original Scene")
    if inserted_xyz.shape[0] > 0:
        ax_bev.scatter(inserted_xyz[:, 0], inserted_xyz[:, 1], s=15, c="red", label="Inserted Object", zorder=3)
        
        # Zoom in on the inserted object for clear inspection
        center = inserted_xyz[:, :2].mean(0)
        ax_bev.set_xlim(center[0] - 20, center[0] + 20)
        ax_bev.set_ylim(center[1] - 20, center[1] + 20)

    ax_bev.set_title("Bird's-Eye-View (Zoomed on Insertion)")
    ax_bev.set_xlabel("X [m] (LiDAR Frame)")
    ax_bev.set_ylabel("Y [m] (LiDAR Frame)")
    ax_bev.set_aspect("equal")
    ax_bev.legend()
    ax_bev.grid(True, linestyle='--', alpha=0.6)

    # Plot 2: Camera Overlay
    img = np.array(Image.open(image_path).convert("RGB"))
    h, w = img.shape[:2]
    ax_cam.imshow(img)
    ax_cam.set_title("Camera Overlay")
    ax_cam.set_axis_off()

    pix_ins = project_velo_to_image(inserted_xyz, calib)
    
    ok_i = ~np.isnan(pix_ins).any(1)
    ax_cam.scatter(pix_ins[ok_i, 0], pix_ins[ok_i, 1], s=10, c="red", edgecolors='white', lw=0.5)

    if inserted_xyz.shape[0] > 0:
        corners_velo = make_axis_aligned_bbox(inserted_xyz)
        corners_img = project_velo_to_image(corners_velo, calib)
        _draw_3d_bbox(ax_cam, corners_img, color="lime", lw=1.5)

    # Save the combined figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Validation plot saved to: {save_path.name}")