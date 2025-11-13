import os
import numpy as np
import SimpleITK as sitk
import glob
from concurrent.futures import ProcessPoolExecutor

# --- Configuration ---

# This map is now FIXED to match your model_architecture.png
TARGET_ORGAN_MAP = {
    "prostate": "prostate.nii.gz",
    "bladder": "bladder.nii.gz",
    "rectum": "rectum.nii.gz"
}
TARGET_ORGAN_NAMES = list(TARGET_ORGAN_MAP.values())
NUM_CLASSES = len(TARGET_ORGAN_MAP)

# Define the target isotropic spacing (in mm) for resampling
TARGET_SPACING = (1.0, 1.0, 1.0)

# Define the fixed output shape (D, H, W)
TARGET_SHAPE = (128, 128, 128)

# Define output directory for processed .npz files
OUTPUT_DIR = "./processed_data"

# --- NEW Helper Function ---

def pad_or_crop_array(array, target_shape, is_mask=False):
    """
    Pad or crop a 3D array to a target shape.
    Performs center-cropping or zero-padding.
    Args:
        array (np.ndarray): The input array (D, H, W).
        target_shape (tuple): The target shape (D, H, W).
        is_mask (bool): If True, use constant padding value 0.
    Returns:
        np.ndarray: The padded or cropped array.
    """
    original_shape = array.shape
    
    # 1. Calculate padding or cropping for each dimension
    # (pad_before, pad_after) or (crop_before, crop_after)
    deltas = []
    for i in range(3):
        delta = target_shape[i] - original_shape[i]
        if delta > 0: # Need padding
            pad_before = delta // 2
            pad_after = delta - pad_before
            deltas.append((pad_before, pad_after))
        else: # Need cropping
            crop_before = abs(delta) // 2
            crop_after = abs(delta) - crop_before
            # We store as negative for slicing
            deltas.append((-crop_before, -crop_after))

    # 2. Apply padding
    if all(d[0] >= 0 and d[1] >= 0 for d in deltas):
        pad_width = deltas
        if is_mask:
            padded_array = np.pad(array, pad_width, mode='constant', constant_values=0)
        else:
            # Pad with minimum value (background)
            padded_array = np.pad(array, pad_width, mode='constant', constant_values=array.min())
        return padded_array
    
    # 3. Apply cropping
    if all(d[0] <= 0 and d[1] <= 0 for d in deltas):
        (d_crop, h_crop, w_crop) = deltas
        d_slice = slice(abs(d_crop[0]), original_shape[0] - abs(d_crop[1]))
        h_slice = slice(abs(h_crop[0]), original_shape[1] - abs(h_crop[1]))
        w_slice = slice(abs(w_crop[0]), original_shape[2] - abs(w_crop[1]))
        return array[d_slice, h_slice, w_slice]

    # 4. Apply mixed padding and cropping (should be rare with resampling)
    # This logic handles cases where one dim is too big and another is too small
    
    # First, pad dimensions that are too small
    pad_width = []
    for i in range(3):
        if deltas[i][0] > 0 or deltas[i][1] > 0:
            pad_width.append((deltas[i][0] if deltas[i][0] > 0 else 0,
                              deltas[i][1] if deltas[i][1] > 0 else 0))
        else:
            pad_width.append((0, 0))
            
    if is_mask:
        padded_array = np.pad(array, pad_width, mode='constant', constant_values=0)
    else:
        padded_array = np.pad(array, pad_width, mode='constant', constant_values=array.min())

    # Second, crop dimensions that are too large
    original_shape = padded_array.shape # New shape after padding
    crop_slices = []
    for i in range(3):
        if deltas[i][0] < 0 or deltas[i][1] < 0:
            crop_before = abs(deltas[i][0]) if deltas[i][0] < 0 else 0
            crop_after = abs(deltas[i][1]) if deltas[i][1] < 0 else 0
            crop_slices.append(slice(crop_before, original_shape[i] - crop_after))
        else:
            crop_slices.append(slice(None)) # No crop
            
    return padded_array[tuple(crop_slices)]


# --- Core Functions ---

def resample_image(itk_image, new_spacing, interpolator):
    """
    Resamples an ITK image to a new spacing.
    """
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    
    # Calculate new size based on new spacing
    new_size = [
        int(round(orig_sz * (orig_sp / new_sp)))
        for orig_sz, orig_sp, new_sp in zip(original_size, original_spacing, new_spacing)
    ]

    # Set up the resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(itk_image.GetDirection())
    resampler.SetOutputOrigin(itk_image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    resampler.SetInterpolator(interpolator)

    return resampler.Execute(itk_image)

def process_patient(patient_dir, output_dir):
    """
    Loads, preprocesses, and saves data for a single patient from the Nifti dataset.
    """
    patient_id = os.path.basename(patient_dir)
    print(f"Processing {patient_id}...")

    # --- 1. Define File Paths ---
    ct_path = os.path.join(patient_dir, "ct.nii.gz")
    seg_dir = os.path.join(patient_dir, "segmentations")
    output_path = os.path.join(output_dir, f"{patient_id}.npz")

    # Check if all required files exist
    if not os.path.exists(ct_path):
        print(f"  [Warning] CT file not found: {ct_path}. Skipping patient.")
        return False
        
    if not os.path.exists(seg_dir):
        print(f"  [Warning] Segmentations directory not found: {seg_dir}. Skipping patient.")
        return False

    try:
        # --- 2. Load and Resample CT ---
        ct_image = sitk.ReadImage(ct_path, sitk.sitkFloat32) 
        resampled_ct_itk = resample_image(ct_image, TARGET_SPACING, sitk.sitkLinear)
        ct_np = sitk.GetArrayFromImage(resampled_ct_itk) # Shape (D, H, W)
        
        # --- 3. Normalize CT (Hounsfield Unit) ---
        ct_np = np.clip(ct_np, -1024.0, 1024.0)
        ct_np = (ct_np - (-1024.0)) / (1024.0 - (-1024.0)) # Normalize to [0, 1]

        # --- 4. Load and Resample Masks ---
        resampled_masks_list = []
        
        for mask_file_name in TARGET_ORGAN_NAMES:
            mask_path = os.path.join(seg_dir, mask_file_name)
            
            if os.path.exists(mask_path):
                mask_image = sitk.ReadImage(mask_path, sitk.sitkUInt8) 
                resampled_mask_itk = resample_image(mask_image, TARGET_SPACING, sitk.sitkNearestNeighbor)
                mask_np = sitk.GetArrayFromImage(resampled_mask_itk) # Shape (D, H, W)
                resampled_masks_list.append(mask_np)
            else:
                print(f"  [Warning] Mask not found: {mask_file_name} in {seg_dir}. Creating empty mask.")
                empty_mask = np.zeros_like(ct_np, dtype=np.uint8)
                resampled_masks_list.append(empty_mask)
        
        # --- 5. NEW: Pad or Crop all arrays to TARGET_SHAPE ---
        ct_np = pad_or_crop_array(ct_np, TARGET_SHAPE, is_mask=False)
        
        processed_masks = []
        for mask_np in resampled_masks_list:
            padded_mask = pad_or_crop_array(mask_np, TARGET_SHAPE, is_mask=True)
            processed_masks.append(padded_mask)

        # --- 6. Stack Masks ---
        # Output shape will be (128, 128, 128, 3)
        masks_np = np.stack(processed_masks, axis=-1)
        masks_np = masks_np.astype(np.uint8)

        # --- 7. Add Channel Dimension to CT ---
        # Reshape CT from (128, 128, 128) to (128, 128, 128, 1)
        ct_np = np.expand_dims(ct_np, axis=-1).astype(np.float32)

        # --- 8. Save Processed Data ---
        np.savez_compressed(output_path, ct=ct_np, mask=masks_np)
        
        print(f"  [Success] Saved {patient_id} to {output_path} with shape {ct_np.shape}")
        return True

    except Exception as e:
        print(f"  [ERROR] Failed to process {patient_id}: {e}")
        return False

def main(data_root):
    """
    Main function to find all patient directories and process them in parallel.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    patient_dirs = sorted(glob.glob(os.path.join(data_root, "BDMAP_*")))
    
    if not patient_dirs:
        print(f"Error: No patient directories found in {data_root}. "
              "Please check the path.")
        return

    print(f"Found {len(patient_dirs)} patient directories.")
    print(f"Target organs: {list(TARGET_ORGAN_MAP.keys())}")
    print(f"Target spacing: {TARGET_SPACING}")
    print(f"Target shape: {TARGET_SHAPE}")
    
    print("Running in serial (one-by-one) to conserve RAM...")
    for p_dir in patient_dirs:
        process_patient(p_dir, OUTPUT_DIR)

    print("\nPreprocessing complete.")
    print(f"Processed data saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    DATA_ROOT_DIR = "/home/yogi/Downloads/AbdomenAtlas1.1Mini_BDMAP_00005001_00005195"
    
    if not os.path.exists(DATA_ROOT_DIR):
        print("="*50)
        print(f"ERROR: Directory not found at {DATA_ROOT_DIR}")
        print("Please double-check the path in preprocess.py")
        print("="*50)
    else:
        main(data_root=DATA_ROOT_DIR)
