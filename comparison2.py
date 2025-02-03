import nibabel as nib
import matplotlib.pyplot as plt
import os
import numpy as np

# Define correct paths
INPUT_FILE = "/root/brainextractor/samples/ds005299/chris_t1.nii"
OUTPUT_FILE = "/root/brainextractor/samples/ds005299/chris_t1output.nii"
BASE_DIR = "/root/brainextractor/slices_two"  # Changed to slices_two

def save_slices(nifti_file, output_dir, num_slices=15, title_prefix=""):
    # Load image
    img = nib.load(nifti_file)
    data = img.get_fdata()
    
    # Create output directories for each plane
    os.makedirs(os.path.join(output_dir, 'axial'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'sagittal'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'coronal'), exist_ok=True)
    
    # Get dimensions
    x_dim, y_dim, z_dim = data.shape
    
    # Calculate slice positions for each dimension
    axial_positions = np.linspace(0, z_dim-1, num_slices, dtype=int)
    sagittal_positions = np.linspace(0, x_dim-1, num_slices, dtype=int)
    coronal_positions = np.linspace(0, y_dim-1, num_slices, dtype=int)
    
    # Generate Axial Slices (Top-Down View)
    for i, pos in enumerate(axial_positions):
        plt.figure(figsize=(10, 10))
        plt.imshow(data[:, :, pos], cmap='gray')
        plt.title(f'{title_prefix}\nAxial Slice {i+1}/{num_slices} (Z={pos})')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'axial', f'axial_slice_{i+1}.png'))
        plt.close()

    # Generate Sagittal Slices (Side View)
    for i, pos in enumerate(sagittal_positions):
        plt.figure(figsize=(10, 10))
        plt.imshow(data[pos, :, :].T, cmap='gray')
        plt.title(f'{title_prefix}\nSagittal Slice {i+1}/{num_slices} (X={pos})')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'sagittal', f'sagittal_slice_{i+1}.png'))
        plt.close()

    # Generate Coronal Slices (Front-Back View)
    for i, pos in enumerate(coronal_positions):
        plt.figure(figsize=(10, 10))
        plt.imshow(data[:, pos, :].T, cmap='gray')
        plt.title(f'{title_prefix}\nCoronal Slice {i+1}/{num_slices} (Y={pos})')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'coronal', f'coronal_slice_{i+1}.png'))
        plt.close()

def process_images():
    # Create main directories
    input_dir = os.path.join(BASE_DIR, 'input_slices')
    output_dir = os.path.join(BASE_DIR, 'output_slices')
    
    # Create directories if they don't exist
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate slices for input image
    print("Generating slices for input image...")
    save_slices(INPUT_FILE, input_dir, title_prefix="Original Brain")
    
    # Generate slices for output image
    print("Generating slices for output image...")
    save_slices(OUTPUT_FILE, output_dir, title_prefix="Extracted Brain")
    
    print("\nSlices have been generated in the following directories:")
    print(f"Input slices: {input_dir}")
    print(f"Output slices: {output_dir}")
    print("\nDirectory structure:")
    print(f"""
    {BASE_DIR}/
    ├── input_slices/
    │   ├── axial/
    │   │   ├── axial_slice_1.png
    │   │   ├── axial_slice_2.png
    │   │   └── ...
    │   ├── sagittal/
    │   │   ├── sagittal_slice_1.png
    │   │   ├── sagittal_slice_2.png
    │   │   └── ...
    │   └── coronal/
    │       ├── coronal_slice_1.png
    │       ├── coronal_slice_2.png
    │       └── ...
    └── output_slices/
        ├── axial/
        │   ├── axial_slice_1.png
        │   ├── axial_slice_2.png
        │   └── ...
        ├── sagittal/
        │   ├── sagittal_slice_1.png
        │   ├── sagittal_slice_2.png
        │   └── ...
        └── coronal/
            ├── coronal_slice_1.png
            ├── coronal_slice_2.png
            └── ...
    """)

if __name__ == "__main__":
    process_images()
