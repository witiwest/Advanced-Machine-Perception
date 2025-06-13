from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from augmentation import _corners_2d, Box

CLASS_COLOR_MAP = {
    'Car': 'red',
    'Pedestrian': 'lime',
    'Cyclist': 'cyan',
    'Other': 'magenta'
}
CLASSES = ['Car', 'Pedestrian', 'Cyclist', 'Other']

# Geometry and Projection Helpers

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

def get_3d_corners_from_box(box: Box) -> np.ndarray:
    """Calculates the 8 3D corners of a Box object in the LiDAR frame."""
    # Get the 2D corners of the rotated box footprint
    bev_corners = _corners_2d(box) # Uses the helper we already have

    # Get the min and max Z values from the box's center and height
    z_min = box.z - (box.h / 2)
    z_max = box.z + (box.h / 2)

    # Create the 8 3D corners
    bottom_corners = np.hstack([bev_corners, np.full((4, 1), z_min)])
    top_corners = np.hstack([bev_corners, np.full((4, 1), z_max)])

    return np.vstack([bottom_corners, top_corners])

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
    inserted_objects_info: list,
    image_path: Path,
    calib: dict,
    save_path: Path
):
    """
    Generates a larger, clearer, and better-zoomed validation image.
    """
    class_names = [info['label'].split()[0] for info in inserted_objects_info]
    title_str = f"Pasted: {', '.join(class_names) or '0 objects'}"

    # --- CHANGE 1: Larger Figure and Better Layout ---
    fig, (ax_bev, ax_cam) = plt.subplots(
        1, 2, 
        figsize=(24, 9), # Increased from (22, 7)
        gridspec_kw={'width_ratios': [1, 1.7]} # Give camera overlay more horizontal space
    )
    fig.suptitle(f"Validation for Frame: {image_path.stem} | {title_str}", fontsize=16)

    # --- Plot 1: Bird's-Eye-View (BEV) ---
    # CHANGE 3: Increased point size for original scene
    ax_bev.scatter(original_xyz[:, 0], original_xyz[:, 1], s=2, c="royalblue", alpha=0.6, label="Original Scene")
    
    if inserted_objects_info:
        all_inserted_pts = np.vstack([info['pts'][:,:3] for info in inserted_objects_info])
        # Plot each inserted object with its own color and larger points
        for info in inserted_objects_info:
            points, box, cls_name = info['pts'], info['box'], info['label'].split()[0]
            color = CLASS_COLOR_MAP.get(cls_name, 'magenta')
            # CHANGE 3: Increased point size for inserted objects
            ax_bev.scatter(points[:, 0], points[:, 1], s=20, c=color, label=cls_name, zorder=10)
            corners = _corners_2d(box)
            ax_bev.plot(np.append(corners[:, 0], corners[0, 0]), np.append(corners[:, 1], corners[0, 1]), c=color, lw=2.0)

        # --- CHANGE 2: Tighter Zoom on BEV Plot ---
        # Calculate a tight window around all inserted objects
        center = all_inserted_pts[:, :2].mean(0)
        # Create a fixed 40m x 40m window for a consistent, focused view
        zoom_size = 40.0 
        ax_bev.set_xlim(center[0] - zoom_size / 2, center[0] + zoom_size / 2)
        ax_bev.set_ylim(center[1] - zoom_size / 2, center[1] + zoom_size / 2)
        # --- END OF ZOOM LOGIC ---

    ax_bev.set_title("Bird's-Eye-View (Zoomed on Insertion)")
    ax_bev.set_xlabel("X [m] (LiDAR Frame)"); ax_bev.set_ylabel("Y [m] (LiDAR Frame)")
    ax_bev.set_aspect("equal"); ax_bev.legend(); ax_bev.grid(True, linestyle='--', alpha=0.6)

    # --- Plot 2: Camera Overlay ---
    img = np.array(Image.open(image_path).convert("RGB"))
    ax_cam.imshow(img); ax_cam.set_title("Camera Overlay"); ax_cam.set_axis_off()

    for info in inserted_objects_info:
        pts, box, cls_name = info['pts'], info['box'], info['label'].split()[0]
        color = CLASS_COLOR_MAP.get(cls_name, 'magenta')
        
        # Project points (with larger size)
        pix_ins = project_velo_to_image(pts[:, :3], calib)
        ok_i = ~np.isnan(pix_ins).any(1)
        # CHANGE 3: Increased point size
        ax_cam.scatter(pix_ins[ok_i, 0], pix_ins[ok_i, 1], s=12, c=color, edgecolors='black', lw=0.5, zorder=10)

        # Project the TRUE 3D box corners
        corners_3d_velo = get_3d_corners_from_box(box)
        corners_img = project_velo_to_image(corners_3d_velo, calib)
        _draw_3d_bbox(ax_cam, corners_img, color=color, lw=2.0)

    # --- CHANGE 1 (Continued): Higher resolution output ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0.1) # Increased dpi
    plt.close(fig)
    print(f"Validation plot saved to: {save_path.name}")