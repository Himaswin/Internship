# Import necessary libraries
import ants  # For brain extraction and image processing
import os
import subprocess
import re
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import nibabel as nib  # For medical image handling
import matplotlib.pyplot as plt
from nilearn import plotting  # For brain visualization
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def convert_numpy_types(obj):
    """
    Convert numpy data types to Python native types for JSON serialization
    Args:
        obj: Input object that might contain numpy data types
    Returns:
        Object with numpy types converted to Python native types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def run_brain_extraction(input_file, output_file):
    """
    Run brain extraction on an input image
    Args:
        input_file: Path to input NIfTI file
        output_file: Path to save extracted brain
    Returns:
        stdout from brain extraction process
    """
    cmd = f"brainextractor {input_file} {output_file}"
    try:
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        return stdout.decode()
    except Exception as e:
        print(f"Error running brain extraction: {str(e)}")
        return None

def parse_brain_extraction_output(output):
    """
    Parse the output text from brain extraction to extract parameters
    Args:
        output: Text output from brain extraction process
    Returns:
        Dictionary containing parsed parameters
    """
    params = {
        'initialization_parameters': {},
        'threshold_values': {},
        'geometric_parameters': {},
        'processing_info': {}
    }
    
    # Regular expressions for parameter extraction
    init_patterns = {
        'bt': r'bt=([\d.]+)',
        'd1': r'd1=([\d.]+)',
        'd2': r'd2=([\d.]+)',
        'rmin': r'rmin=([\d.]+)',
        'rmax': r'rmax=([\d.]+)'
    }
    
    threshold_patterns = {
        'tmin': r'tmin: ([\d.]+)',
        't2': r't2: ([\d.]+)',
        't': r't: ([\d.]+)',
        't98': r't98: ([\d.]+)',
        'tmax': r'tmax: ([\d.]+)'
    }

    # Extract initialization parameters
    for param_name, pattern in init_patterns.items():
        match = re.search(pattern, output)
        if match:
            params['initialization_parameters'][param_name] = float(match.group(1))

    # Extract threshold values
    for param_name, pattern in threshold_patterns.items():
        match = re.search(pattern, output)
        if match:
            params['threshold_values'][param_name] = float(match.group(1))

    # Extract geometric parameters
    com_match = re.search(r'Center-of-Mass: $$(.*?)$$', output)
    if com_match:
        com_values = [float(x) for x in com_match.group(1).split()]
        params['geometric_parameters']['center_of_mass'] = com_values

    radius_match = re.search(r'Head Radius: ([\d.]+)', output)
    if radius_match:
        params['geometric_parameters']['head_radius'] = float(radius_match.group(1))

    median_match = re.search(r'Median within Head Radius: ([\d.]+)', output)
    if median_match:
        params['geometric_parameters']['median'] = float(median_match.group(1))

    # Extract processing information
    iter_match = re.search(r'Iteration: (\d+)', output)
    if iter_match:
        params['processing_info']['iterations'] = int(iter_match.group(1))

    return params

def calculate_differences(original, defaced):
    """
    Calculate differences between original and defaced parameters
    Args:
        original: Parameter value from original image
        defaced: Parameter value from defaced image
    Returns:
        Dictionary containing absolute and percentage differences
    """
    if isinstance(original, (int, float)) and isinstance(defaced, (int, float)):
        # For single numerical values
        abs_diff = abs(original - defaced)
        perc_diff = (abs_diff / original * 100) if original != 0 else 0
        return {
            'absolute_difference': float(abs_diff),
            'percentage_difference': float(perc_diff)
        }
    elif isinstance(original, list) and isinstance(defaced, list):
        # For vector values (like center-of-mass)
        euclidean_dist = float(np.sqrt(sum((o - d) ** 2 for o, d in zip(original, defaced))))
        orig_norm = float(np.linalg.norm(original))
        def_norm = float(np.linalg.norm(defaced))
        perc_diff = float(abs(orig_norm - def_norm) / orig_norm * 100 if orig_norm != 0 else 0)
        return {
            'euclidean_distance': euclidean_dist,
            'percentage_difference': perc_diff
        }
    return None

def compare_parameters(original_params, defaced_params):
    """
    Compare parameters between original and defaced brain extractions
    Args:
        original_params: Parameters from original image
        defaced_params: Parameters from defaced image
    Returns:
        Dictionary containing comparison results and metrics
    """
    comparison_results = {
        'initialization_parameters': {},
        'threshold_values': {},
        'geometric_parameters': {},
        'processing_info': {},
        'overall_metrics': {}
    }

    def compare_group(orig_group, def_group, result_group):
        """
        Compare a group of parameters and calculate differences
        """
        total_diff_percentage = 0
        num_params = 0
        
        for param_name in orig_group:
            if param_name in def_group:
                result_group[param_name] = {
                    'original': convert_numpy_types(orig_group[param_name]),
                    'defaced': convert_numpy_types(def_group[param_name])
                }
                
                differences = calculate_differences(orig_group[param_name], def_group[param_name])
                if differences:
                    result_group[param_name].update(differences)
                    if 'percentage_difference' in differences:
                        total_diff_percentage += differences['percentage_difference']
                        num_params += 1
        
        return float(total_diff_percentage), num_params

    # Compare each parameter group
    total_diff = 0
    total_params = 0
    
    for group in ['initialization_parameters', 'threshold_values', 'geometric_parameters', 'processing_info']:
        diff, num = compare_group(
            original_params[group],
            defaced_params[group],
            comparison_results[group]
        )
        total_diff += diff
        total_params += num

    # Calculate overall metrics
    if total_params > 0:
        overall_similarity = float(100 - (total_diff / total_params))
        comparison_results['overall_metrics']['similarity_score'] = overall_similarity
        comparison_results['overall_metrics']['total_parameters_compared'] = total_params
        comparison_results['overall_metrics']['average_difference_percentage'] = float(total_diff / total_params)

    return comparison_results

def visualize_differences(original_file, defaced_file, output_dir):
    """
    Generate visualizations comparing original and defaced brain regions
    Args:
        original_file: Path to original NIfTI file
        defaced_file: Path to defaced NIfTI file
        output_dir: Directory to save visualization outputs
    """
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Load images
        orig_img = nib.load(original_file)
        def_img = nib.load(defaced_file)
        orig_seg = nib.load(f"{output_dir}/original_first_all_fast_firstseg.nii.gz")
        def_seg = nib.load(f"{output_dir}/defaced_first_all_fast_firstseg.nii.gz")

        orig_data = orig_seg.get_fdata()
        def_data = def_seg.get_fdata()

        # Define brain regions to compare
        brain_regions = {
            'Left_Amygdala': {'label': 18},
            'Right_Amygdala': {'label': 54},
            'Left_Hippocampus': {'label': 17},
            'Right_Hippocampus': {'label': 53},
            'Left_Thalamus': {'label': 10},
            'Right_Thalamus': {'label': 49},
        }

        # Generate visualizations for each brain region
        for region_name, info in brain_regions.items():
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            display_modes = ['x', 'y', 'z']

            for mode_idx, mode in enumerate(display_modes):
                ax = axes[mode_idx]
                orig_mask = orig_data == info['label']
                def_mask = def_data == info['label']

                # Calculate differences
                a_minus_b = orig_mask & ~def_mask  # In original but not in defaced
                b_minus_a = def_mask & ~orig_mask  # In defaced but not in original
                a_and_b = orig_mask & def_mask     # In both

                # Create combined visualization
                combined_img = np.zeros(orig_mask.shape)
                combined_img[a_minus_b] = 1  # Red - removed by defacing
                combined_img[b_minus_a] = 2  # Blue - added after defacing
                combined_img[a_and_b] = 3    # Grey - unchanged

                # Plot the differences
                display = plotting.plot_anat(
                    nib.Nifti1Image(combined_img, orig_img.affine),
                    display_mode=mode,
                    cut_coords=1,
                    title=f'{region_name} - {mode.upper()} view',
                    axes=ax,
                    cmap=plt.cm.colors.ListedColormap(['black', 'red', 'blue', 'grey']),
                    vmin=0, vmax=3
                )

            plt.tight_layout()
            diff_file = f"{output_dir}/{region_name}_differences.png"
            plt.savefig(diff_file)
            plt.close()

        print(f"\nGenerated difference files:")
        for region_name in brain_regions.keys():
            print(f"- {region_name} differences: {output_dir}/{region_name}_differences.png")

    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        import traceback
        traceback.print_exc()

def find_matching_files(input_dir, defaced_dir):
    """
    Find matching pairs of original and defaced files
    Args:
        input_dir: Directory containing original files
        defaced_dir: Directory containing defaced files
    Returns:
        Tuple of (matched_pairs, unmatched_files)
    """
    matched_pairs = []
    unmatched_originals = []
    
    # Get all .nii.gz files from both directories
    input_files = [f for f in os.listdir(input_dir) if f.endswith('.nii.gz')]
    defaced_files = [f for f in os.listdir(defaced_dir) if f.endswith('.nii.gz')]
    
    # Create a dictionary of defaced files for faster lookup
    defaced_dict = {
        f.replace('_defaced', ''): f  # Remove '_defaced' suffix if present
        for f in defaced_files
    }
    
    for input_file in input_files:
        original_path = os.path.join(input_dir, input_file)
        
        # Try different possible matching patterns
        base_name = input_file.replace('.nii.gz', '')
        possible_matches = [
            input_file,  # Exact match
            f"{base_name}_defaced.nii.gz",  # With _defaced suffix
            f"defaced_{base_name}.nii.gz",  # With defaced_ prefix
        ]
        
        matched = False
        for possible_match in possible_matches:
            if possible_match in defaced_files:
                defaced_path = os.path.join(defaced_dir, possible_match)
                matched_pairs.append((original_path, defaced_path))
                matched = True
                break
        
        if not matched:
            unmatched_originals.append(input_file)
    
    return matched_pairs, unmatched_originals

def print_comparison_results(results):
    """
    Print formatted comparison results
    Args:
        results: Dictionary containing comparison results
    """
    print("\nBrain Extraction Comparison Results:")
    print("=" * 80)

    def print_parameter_group(group_name, group_data):
        print(f"\n{group_name}:")
        print("-" * 40)
        for param_name, values in group_data.items():
            print(f"\n  {param_name.replace('_', ' ').title()}:")
            for key, value in values.items():
                if isinstance(value, list):
                    print(f"    {key}: [{', '.join([f'{x:.3f}' for x in value])}]")
                elif isinstance(value, float):
                    print(f"    {key}: {value:.3f}")
                else:
                    print(f"    {key}: {value}")

    # Print each parameter group
    for group in ['initialization_parameters', 'threshold_values', 
                 'geometric_parameters', 'processing_info']:
        print_parameter_group(group.replace('_', ' ').title(), results[group])

    # Print overall metrics
    print("\nOverall Metrics:")
    print("-" * 40)
    for metric, value in results['overall_metrics'].items():
        print(f"  {metric.replace('_', ' ').title()}: {value:.2f}")

def save_results_to_file(results, filename):
    """
    Save comparison results to JSON file
    Args:
        results: Dictionary containing results
        filename: Path to save JSON file
    """
    try:
        converted_results = convert_numpy_types(results)
        with open(filename, 'w') as f:
            json.dump(converted_results, f, indent=4)
        print(f"\nResults successfully saved to: {filename}")
    except Exception as e:
        print(f"Error saving results to file: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """
    Main execution function
    """
    # Define directories
    input_dir = "/root/def_bio/testingfiles/input_check"
    defaced_dir = "/root/def_bio/testingfiles/output_fsl_deface"
    output_base_dir = "/root/def_bio/testingfiles/comparison_results"
    
    # Create main output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Create a log file
    log_file = os.path.join(output_base_dir, 
                           f"comparison_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    def log_message(message):
        """Local function to handle logging"""
        print(message)
        with open(log_file, 'a') as f:
            f.write(f"{message}\n")

    # Find matching files
    matched_pairs, unmatched = find_matching_files(input_dir, defaced_dir)
    
    log_message(f"\nStarting comparison at {datetime.now()}")
    log_message(f"Found {len(matched_pairs)} matching pairs")
    if unmatched:
        log_message(f"Warning: {len(unmatched)} files without matching defaced versions:")
        for f in unmatched:
            log_message(f"  - {f}")
    
    # Process each matched pair
    for idx, (original_input, defaced_input) in enumerate(matched_pairs, 1):
        try:
            log_message(f"\nProcessing pair {idx}/{len(matched_pairs)}")
            log_message(f"Original: {os.path.basename(original_input)}")
            log_message(f"Defaced:  {os.path.basename(defaced_input)}")
            
            # Create directory structure for this comparison
            base_name = os.path.basename(original_input).replace('.nii.gz', '')
            file_output_dir = os.path.join(output_base_dir, base_name)
            
            # Create subdirectories
            extraction_dir = os.path.join(file_output_dir, 'extractions')
            visualization_dir = os.path.join(file_output_dir, 'visualizations')
            analysis_dir = os.path.join(file_output_dir, 'analysis')
            
            for dir_path in [extraction_dir, visualization_dir, analysis_dir]:
                os.makedirs(dir_path, exist_ok=True)
            
            # Define output paths
            original_output = os.path.join(extraction_dir, "original_extraction.nii")
            defaced_output = os.path.join(extraction_dir, "defaced_extraction.nii")
            
            # Run brain extractions
            log_message("Running brain extractions...")
            original_output_text = run_brain_extraction(original_input, original_output)
            defaced_output_text = run_brain_extraction(defaced_input, defaced_output)
            
            if original_output_text and defaced_output_text:
                # Parse and compare parameters
                original_params = parse_brain_extraction_output(original_output_text)
                defaced_params = parse_brain_extraction_output(defaced_output_text)
                
                # Generate results dictionary
                results = {
                    'parameter_comparison': compare_parameters(original_params, defaced_params),
                    'metadata': {
                        'original_file': os.path.basename(original_input),
                        'defaced_file': os.path.basename(defaced_input),
                        'processing_date': datetime.now().isoformat(),
                        'extraction_paths': {
                            'original': original_output,
                            'defaced': defaced_output
                        }
                    }
                }
                
                # Save comparison results
                results_file = os.path.join(analysis_dir, 
                    f"comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                save_results_to_file(results, results_file)
                
                # Generate visualizations
                log_message("Generating visualizations...")
                visualize_differences(original_input, defaced_input, visualization_dir)
                
                # Track generated files
                generated_files = {
                    'Extractions': [original_output, defaced_output],
                    'Analysis': [results_file],
                    'Visualizations': [
                        os.path.join(visualization_dir, f) 
                        for f in os.listdir(visualization_dir) 
                        if f.endswith('.png')
                    ]
                }
                
                # Save file inventory
                inventory_file = os.path.join(file_output_dir, 'file_inventory.json')
                save_results_to_file(generated_files, inventory_file)
                
                # Log generated files
                log_message("\nGenerated files:")
                for category, files in generated_files.items():
                    log_message(f"\n{category}:")
                    for f in files:
                        log_message(f"  - {os.path.basename(f)}")
                
            else:
                log_message(f"Error: Failed to run brain extractions for {base_name}")
                
        except Exception as e:
            log_message(f"Error processing pair: {str(e)}")
            import traceback
            log_message(traceback.format_exc())
            continue
    
    log_message("\nProcessing complete!")

if __name__ == "__main__":
    main()
