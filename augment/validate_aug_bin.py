import numpy as np
import sys
import os
import glob
from pathlib import Path

def validate_point_painting_format(file_path: str):
    """
    Loads an augmented .bin file and runs rigorous checks to ensure it's
    ready for the PointPainting pipeline.
    """
    print(f"Validating file: {Path(file_path).name}")

    try:
        # 1. Load the binary point cloud data
        pc = np.fromfile(file_path, dtype=np.float32)
        
        # 2. Check for correct dimensionality (must be 9)
        assert pc.size % 9 == 0, f"File size is not divisible by 9. Something is wrong with the data format."
        pc = pc.reshape(-1, 9)
        print(f"PASSED: Point cloud has correct shape: {pc.shape}")

        # 3. Check data type
        assert pc.dtype == np.float32, f"Data type is not float32. Found {pc.dtype}."
        print(f"PASSED: Data type is correct: {pc.dtype}")

        # 4. Check that the semantic channels (last 5) are normalized (sum to 1.0)
        semantic_channels = pc[:, 4:]
        sums = np.sum(semantic_channels, axis=1)
        assert np.allclose(sums, 1.0), "Semantic channels are not normalized (sum is not 1.0 for all points)."
        print(f"PASSED: Semantic channels are correctly normalized.")

        # 5. Check for data integrity: a point is either "unknown" or an object, not a mix.
        is_unknown = semantic_channels[:, 0] == 1.0
        is_object = semantic_channels[:, 0] == 0.0
        object_class_sums = np.sum(semantic_channels[is_object, 1:], axis=1)
        assert np.allclose(object_class_sums, 1.0), "Object points have mixed or zero classes."
        print(f"PASSED: Data integrity check (no mixed classes).")

        print(f"[SUCCESS] File '{Path(file_path).name}' is valid.")
        return True

    except FileNotFoundError:
        print(f"[ERROR] File not found at: {file_path}")
        return False
    except AssertionError as e:
        print(f"\n[FAILED] Validation failed for '{Path(file_path).name}': {e}")
        return False

if __name__ == '__main__':    
    print("Batch validation mode: searching for all augmented .bin files...\n")
    # Define the directory where augmented files are saved
    aug_dir = Path.home() / "final_assignment" / "augmented_frames"
    
    # Find all files ending with _aug.bin in that directory
    files_to_check = sorted(glob.glob(str(aug_dir / "*_aug.bin")))
    
    if not files_to_check:
        print(f"No augmented files ('*_aug.bin') found in directory: {aug_dir}")
        sys.exit(0)
        
    success_count = 0
    fail_count = 0
    
    for file_path in files_to_check:
        if validate_point_painting_format(file_path):
            success_count += 1
        else:
            fail_count += 1
        print("-" * 50)
        
    print("\n--- BATCH VALIDATION COMPLETE ---")
    print(f"Successfully validated: {success_count} file(s)")
    print(f"Failed validation:      {fail_count} file(s)")