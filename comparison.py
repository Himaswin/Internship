import nibabel as nib
import matplotlib.pyplot as plt
import os
import numpy as np

# Define correct paths
INPUT_FILE = "/root/brainextractor/samples/ds005299/chris_t1.nii"
OUTPUT_FILE = "/root/brainextractor/samples/ds005299/chris_t1output.nii"
SLICE_DIR = "/root/brainextractor/slices/comparisons"

def create_comparison_slices(before_file, after_file, output_dir, num_slices=15):
    # Load images
    img_before = nib.load(before_file)
    img_after = nib.load(after_file)
    
    data_before = img_before.get_fdata()
    data_after = img_after.get_fdata()
    
    # Create output directories for each plane
    os.makedirs(os.path.join(output_dir, 'axial'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'sagittal'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'coronal'), exist_ok=True)
    
    # Get dimensions
    x_dim, y_dim, z_dim = data_before.shape
    
    # Calculate slice positions for each dimension
    axial_positions = np.linspace(0, z_dim-1, num_slices, dtype=int)     # Top to bottom
    sagittal_positions = np.linspace(0, x_dim-1, num_slices, dtype=int)  # Left to right
    coronal_positions = np.linspace(0, y_dim-1, num_slices, dtype=int)   # Front to back
    
    # Generate Axial Slices (Top-Down View)
    for i, pos in enumerate(axial_positions):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        ax1.imshow(data_before[:, :, pos], cmap='gray')
        ax1.set_title(f'Before Brain Extraction\nAxial Slice {i+1}/{num_slices} (Z={pos})')
        ax1.axis('off')
        
        ax2.imshow(data_after[:, :, pos], cmap='gray')
        ax2.set_title(f'After Brain Extraction\nAxial Slice {i+1}/{num_slices} (Z={pos})')
        ax2.axis('off')
        
        plt.savefig(os.path.join(output_dir, 'axial', f'axial_slice_{i+1}.png'))
        plt.close()

    # Generate Sagittal Slices (Side View)
    for i, pos in enumerate(sagittal_positions):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        ax1.imshow(data_before[pos, :, :].T, cmap='gray')
        ax1.set_title(f'Before Brain Extraction\nSagittal Slice {i+1}/{num_slices} (X={pos})')
        ax1.axis('off')
        
        ax2.imshow(data_after[pos, :, :].T, cmap='gray')
        ax2.set_title(f'After Brain Extraction\nSagittal Slice {i+1}/{num_slices} (X={pos})')
        ax2.axis('off')
        
        plt.savefig(os.path.join(output_dir, 'sagittal', f'sagittal_slice_{i+1}.png'))
        plt.close()

    # Generate Coronal Slices (Front-Back View)
    for i, pos in enumerate(coronal_positions):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        ax1.imshow(data_before[:, pos, :].T, cmap='gray')
        ax1.set_title(f'Before Brain Extraction\nCoronal Slice {i+1}/{num_slices} (Y={pos})')
        ax1.axis('off')
        
        ax2.imshow(data_after[:, pos, :].T, cmap='gray')
        ax2.set_title(f'After Brain Extraction\nCoronal Slice {i+1}/{num_slices} (Y={pos})')
        ax2.axis('off')
        
        plt.savefig(os.path.join(output_dir, 'coronal', f'coronal_slice_{i+1}.png'))
        plt.close()

# Create comparison slices using the correct file paths
create_comparison_slices(INPUT_FILE, OUTPUT_FILE, SLICE_DIR)

print(f"Comparison slices have been saved to: {SLICE_DIR}")
print("Generated slices in three anatomical planes:")
print("1. Axial (Top-Down view) - in /axial folder")
print("2. Sagittal (Side view) - in /sagittal folder")
print("3. Coronal (Front-Back view) - in /coronal folder")
