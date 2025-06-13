from pathlib import Path
import numpy as np
import os

from augmentation import DataAugmenter, load_kitti_calib
from validation_visualizer import create_validation_plot
from omegaconf import OmegaConf

# We can now test on more frames to see the variety.
FRAMES     = ["00100", "00242", "00376", "00550", "00816", "01100", "01450", "01900", "02300", "03500", "04000", "04500", "05000", "07000", "07500", "08500", "09000"]
OBJ_DICT   = Path.home() / "final_assignment" / "common_src" / "object_dict.pkl"
DATA_ROOT  = Path.home() / "final_assignment" / "data" / "view_of_delft"
AUG_DIR    = Path.home() / "final_assignment" / "augmented_frames"
FIG_DIR    = Path.home() / "final_assignment" / "tests" / "figures"
AUG_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

def save_point_cloud(p, pts): 
    """Saves a point cloud array to a .bin file."""
    pts.astype(np.float32).tofile(p)

def save_labels(p, lines):    
    """Saves a list of label strings to a .txt file."""
    Path(p).write_text("\n".join(l.rstrip() for l in lines)+"\n")


if __name__ == "__main__":
    print("Running in DEBUG mode")

    copy_paste_config = OmegaConf.create({
    'obj_db_path': str(OBJ_DICT),
    'prob': 0.8,  # Always attempt to augment the scene
    'max_trials': 50, # Max attempts to place EACH object

    # --- ADD THIS SECTION TO ENABLE MULTI-OBJECT INSERTION ---
    'multi_object': {
        'enabled': True,
        # The maximum number of objects the script will try to add
        'max_objects': 1,
        # The probability of attempting to add the Nth object.
        # e.g., 100% for 1st, 80% for 2nd, 30% for 3rd.
        'attempt_probs': [0.8]
    }
})

    augmenter = DataAugmenter(cfg=copy_paste_config)

    for frame in FRAMES:
        print(f"\nProcessing frame {frame}...")
        
        # Load a single data sample from disk for this test
        lidar_path = DATA_ROOT / "lidar/training/velodyne" / f"{frame}.bin"
        label_path = DATA_ROOT / "lidar/training/label_2" / f"{frame}.txt"
        calib_path = DATA_ROOT / "lidar/training/calib" / f"{frame}.txt"
        rgb_path   = DATA_ROOT / "lidar/training/image_2" / f"{frame}.jpg"

        pc = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        with open(label_path, 'r') as f:
            labels = f.read().splitlines()
        calib = load_kitti_calib(calib_path)

        # Create the data sample dictionary and pass it to the augmenter
        original_sample = {'pc': pc, 'labels': labels, 'calib': calib}
        augmented_sample, placed_objects_info = augmenter(original_sample)

        if placed_objects_info: # Check if the list is not empty
            num_added = len(placed_objects_info)
            print(f"   ✓ Augmentation successful (inserted {num_added} object(s)).")
            
            # Save the augmented .bin and .txt files
            aug_bin_path = AUG_DIR / f"{frame}_aug.bin"
            aug_label_path = AUG_DIR / f"{frame}_aug.txt"
            save_point_cloud(aug_bin_path, augmented_sample['pc'])
            save_labels(aug_label_path, augmented_sample['labels'])
            print(f"   ✓ Saved augmented data to {aug_bin_path.name} and {aug_label_path.name}")

            # Generate the validation plot using the rich info
            # We no longer need split_cloud because we have the exact objects.
            original_xyz = original_sample['pc'][:, :3]
            
            val_plot_path = FIG_DIR / f"{frame}_validation.png"
            create_validation_plot(
                original_xyz=original_xyz,
                inserted_objects_info=placed_objects_info, # Pass the rich info list
                image_path=rgb_path,
                calib=calib,
                save_path=val_plot_path
            )
        else:
            print("Augmentation attempt failed (no valid pose found).")

    print("\nDebug run finished")