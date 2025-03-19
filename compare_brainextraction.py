import subprocess
import re
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import os
import SimpleITK as sitk
from radiomics import featureextractor
import pandas as pd
import matplotlib.pyplot as plt

# Brain Extraction Functions
def run_brain_extraction(input_file, output_file):
    cmd = f"brainextractor {input_file} {output_file}"
    try:
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        return stdout.decode()
    except Exception as e:
        print(f"Error running brain extraction: {str(e)}")
        return None

def parse_brain_extraction_output(output):
    params = {
        'initialization_parameters': {}, 'threshold_values': {}, 'geometric_parameters': {}, 'processing_info': {}
    }
    init_patterns = {'bt': r'bt=([\d.]+)', 'd1': r'd1=([\d.]+)', 'd2': r'd2=([\d.]+)', 'rmin': r'rmin=([\d.]+)', 'rmax': r'rmax=([\d.]+)'}
    threshold_patterns = {'tmin': r'tmin: ([\d.]+)', 't2': r't2: ([\d.]+)', 't': r't: ([\d.]+)', 't98': r't98: ([\d.]+)', 'tmax': r'tmax: ([\d.]+)'}
    
    for param_name, pattern in init_patterns.items():
        match = re.search(pattern, output)
        if match: params['initialization_parameters'][param_name] = float(match.group(1))
    
    for param_name, pattern in threshold_patterns.items():
        match = re.search(pattern, output)
        if match: params['threshold_values'][param_name] = float(match.group(1))
    
    com_match = re.search(r'Center-of-Mass: $$(.*?)$$', output)
    if com_match: params['geometric_parameters']['center_of_mass'] = [float(x) for x in com_match.group(1).split()]
    
    radius_match = re.search(r'Head Radius: ([\d.]+)', output)
    if radius_match: params['geometric_parameters']['head_radius'] = float(radius_match.group(1))
    
    median_match = re.search(r'Median within Head Radius: ([\d.]+)', output)
    if median_match: params['geometric_parameters']['median'] = float(median_match.group(1))
    
    iter_match = re.search(r'Iteration: (\d+)', output)
    if iter_match: params['processing_info']['iterations'] = int(iter_match.group(1))
    
    return params

def calculate_differences(original, defaced):
    if isinstance(original, (int, float)) and isinstance(defaced, (int, float)):
        abs_diff = abs(original - defaced)
        perc_diff = (abs_diff / original * 100) if original != 0 else 0
        return {'absolute_difference': abs_diff, 'percentage_difference': perc_diff}
    elif isinstance(original, list) and isinstance(defaced, list):
        euclidean_dist = np.sqrt(sum((o - d) ** 2 for o, d in zip(original, defaced)))
        orig_norm = np.linalg.norm(original)
        def_norm = np.linalg.norm(defaced)
        perc_diff = abs(orig_norm - def_norm) / orig_norm * 100 if orig_norm != 0 else 0
        return {'euclidean_distance': euclidean_dist, 'percentage_difference': perc_diff}
    return None

def compare_parameters(original_params, defaced_params):
    comparison_results = {
        'initialization_parameters': {}, 'threshold_values': {}, 'geometric_parameters': {},
        'processing_info': {}, 'overall_metrics': {}
    }
    
    def compare_group(orig_group, def_group, result_group):
        total_diff_percentage = 0
        num_params = 0
        for param_name in orig_group:
            if param_name in def_group:
                result_group[param_name] = {'original': orig_group[param_name], 'defaced': def_group[param_name]}
                differences = calculate_differences(orig_group[param_name], def_group[param_name])
                if differences:
                    result_group[param_name].update(differences)
                    if 'percentage_difference' in differences:
                        total_diff_percentage += differences['percentage_difference']
                        num_params += 1
        return total_diff_percentage, num_params
    
    total_diff = 0
    total_params = 0
    for group in ['initialization_parameters', 'threshold_values', 'geometric_parameters', 'processing_info']:
        diff, num = compare_group(original_params[group], defaced_params[group], comparison_results[group])
        total_diff += diff
        total_params += num
    
    if total_params > 0:
        overall_similarity = 100 - (total_diff / total_params)
        comparison_results['overall_metrics']['similarity_score'] = overall_similarity
        comparison_results['overall_metrics']['total_parameters_compared'] = total_params
        comparison_results['overall_metrics']['average_difference_percentage'] = total_diff / total_params
    
    return comparison_results

def print_comparison_results(results):
    print("\nBrain Extraction Comparison Results:")
    print("=" * 80)
    def print_parameter_group(group_name, group_data):
        print(f"\n{group_name}:")
        print("-" * 40)
        for param_name, values in group_data.items():
            print(f"\n  {param_name.replace('_', ' ').title()}:")
            for key, value in values.items():
                if isinstance(value, list): print(f"    {key}: [{', '.join([f'{x:.3f}' for x in value])}]")
                elif isinstance(value, float): print(f"    {key}: {value:.3f}")
                else: print(f"    {key}: {value}")
    
    for group in ['initialization_parameters', 'threshold_values', 'geometric_parameters', 'processing_info']:
        print_parameter_group(group.replace('_', ' ').title(), results[group])
    print("\nOverall Metrics:")
    print("-" * 40)
    for metric, value in results['overall_metrics'].items():
        print(f"  {metric.replace('_', ' ').title()}: {value:.2f}")

def save_results_to_file(results, filename):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)

# Radiomics Functions
def extract_radiomics_features(image_path, mask_path, output_csv):
    print(f"Extracting radiomics features from {image_path}")
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None
    
    settings = {'binWidth': 25, 'resampledPixelSpacing': None, 'interpolator': sitk.sitkBSpline, 'verbose': True}
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.enableAllFeatures()
    
    if not os.path.exists(mask_path):
        print(f"Mask not found at {mask_path}. Creating simple mask...")
        success = create_simple_mask(image_path, mask_path)
        if not success: return None
    
    try:
        result = extractor.execute(image_path, mask_path)
        feature_df = pd.DataFrame([[key, value] for key, value in result.items() if not isinstance(value, sitk.Image)],
                                 columns=['Feature', 'Value'])
        feature_df.to_csv(output_csv, index=False)
        print(f"Features saved to {output_csv}")
        return feature_df
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def create_simple_mask(image_path, output_mask_path):
    try:
        img = sitk.ReadImage(image_path)
        stats = sitk.StatisticsImageFilter()
        stats.Execute(img)
        threshold = stats.GetMean()
        mask = sitk.BinaryThreshold(img, lowerThreshold=threshold, upperThreshold=float('inf'), insideValue=1, outsideValue=0)
        mask = sitk.BinaryMorphologicalClosing(mask, [3, 3, 3])
        sitk.WriteImage(mask, output_mask_path)
        print(f"Mask created at {output_mask_path}")
        return True
    except Exception as e:
        print(f"Error creating mask: {e}")
        return False

def compare_radiomics_features(initial_df, defaced_df, output_csv, plot_path, region_name=""):
    comparison_data = []
    for feature in initial_df['Feature']:
        if feature in defaced_df['Feature'].values:
            initial_val = initial_df[initial_df['Feature'] == feature]['Value'].iloc[0]
            defaced_val = defaced_df[defaced_df['Feature'] == feature]['Value'].iloc[0]
            if isinstance(initial_val, (int, float)) and isinstance(defaced_val, (int, float)) and initial_val != 0:
                abs_diff = abs(defaced_val - initial_val)
                perc_diff = (abs_diff / initial_val) * 100
                comparison_data.append([feature, initial_val, defaced_val, abs_diff, perc_diff])
    
    # Create DataFrame with detailed data
    comparison_df = pd.DataFrame(comparison_data, columns=['Feature', 'Initial', 'Defaced', 'Absolute_Diff', 'Percent_Diff'])
    comparison_df.to_csv(output_csv, index=False)
    print(f"{region_name}Radiomics comparison data saved to {output_csv}")
    
    # Generate bar plot (unchanged visualization)
    top_diff = comparison_df.sort_values(by='Percent_Diff', key=abs, ascending=False).head(10)
    plt.figure(figsize=(12, 6))
    plt.barh(top_diff['Feature'], top_diff['Percent_Diff'], color='skyblue')
    plt.xlabel('Percent Difference (%)')
    plt.title(f'Top Radiomics Feature Differences: Original vs Defaced ({region_name})')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"{region_name}Comparison plot saved to {plot_path}")
    
    # Print summary statistics
    print(f"\n{region_name}Radiomics Comparison Summary:")
    print(f"Average Absolute Difference: {comparison_df['Absolute_Diff'].mean():.2f}")
    print(f"Average Percent Difference: {comparison_df['Percent_Diff'].mean():.2f}%")
    print(f"Max Percent Difference: {comparison_df['Percent_Diff'].max():.2f}% (Feature: {comparison_df.loc[comparison_df['Percent_Diff'].idxmax(), 'Feature']})")
    
    return comparison_df

def plot_voxel_intensity_comparison(original_image, defaced_image, output_path, region_name=""):
    orig_img = sitk.ReadImage(original_image)
    def_img = sitk.ReadImage(defaced_image)
    orig_array = sitk.GetArrayFromImage(orig_img).flatten()
    def_array = sitk.GetArrayFromImage(def_img).flatten()
    
    plt.figure(figsize=(10, 6))
    plt.hist(orig_array, bins=100, alpha=0.5, label='Original', color='blue')
    plt.hist(def_array, bins=100, alpha=0.5, label='Defaced', color='orange')
    plt.xlabel('Voxel Intensity')
    plt.ylabel('Frequency')
    plt.title(f'Voxel Intensity Distribution: Original vs Defaced ({region_name})')
    plt.legend()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"{region_name}Voxel intensity plot saved to {output_path}")

def main():
    # File paths
    original_input = "/root/def_bio/samples/niivue-images/chris_t1.nii.gz"
    original_output = "/root/def_bio/samples/output/chris_t1output.nii"
    defaced_input = "/root/def_bio/samples/output/chris_t1_defaced.nii.gz"
    defaced_output = "/root/def_bio/samples/output/chris_t1output_defaced_extraction.nii"
    output_dir = "/root/def_bio/samples/output/radiomics_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Assuming amygdala mask files (replace with actual paths if available)
    amygdala_mask_original = f"{output_dir}/amygdala_mask_original.nii"  # Placeholder
    amygdala_mask_defaced = f"{output_dir}/amygdala_mask_defaced.nii"    # Placeholder
    whole_brain_mask_original = f"{output_dir}/whole_brain_mask_original.nii"
    whole_brain_mask_defaced = f"{output_dir}/whole_brain_mask_defaced.nii"

    # Run brain extractions
    print("Running original brain extraction...")
    original_output_text = run_brain_extraction(original_input, original_output)
    print("Running defaced brain extraction...")
    defaced_output_text = run_brain_extraction(defaced_input, defaced_output)

    if original_output_text and defaced_output_text:
        # Parse and compare brain extraction parameters
        original_params = parse_brain_extraction_output(original_output_text)
        defaced_params = parse_brain_extraction_output(defaced_output_text)
        results = compare_parameters(original_params, defaced_params)
        print_comparison_results(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"{output_dir}/brain_extraction_comparison_{timestamp}.json"
        save_results_to_file(results, results_file)
        print(f"Brain extraction results saved to: {results_file}")

        # Extract radiomics features for whole brain
        original_features_csv = f"{output_dir}/original_features.csv"
        defaced_features_csv = f"{output_dir}/defaced_features.csv"
        original_df = extract_radiomics_features(original_output, whole_brain_mask_original, original_features_csv)
        defaced_df = extract_radiomics_features(defaced_output, whole_brain_mask_defaced, defaced_features_csv)

        # Extract radiomics features for amygdala (assuming masks are provided or generated)
        amygdala_original_features_csv = f"{output_dir}/amygdala_original_features.csv"
        amygdala_defaced_features_csv = f"{output_dir}/amygdala_defaced_features.csv"
        amygdala_original_df = extract_radiomics_features(original_output, amygdala_mask_original, amygdala_original_features_csv)
        amygdala_defaced_df = extract_radiomics_features(defaced_output, amygdala_mask_defaced, amygdala_defaced_features_csv)

        # Compare radiomics features and generate outputs
        if original_df is not None and defaced_df is not None:
            # Whole brain comparison
            comparison_csv = f"{output_dir}/whole_brain_comparison_{timestamp}.csv"
            plot_path = f"{output_dir}/whole_brain_comparison_plot_{timestamp}.png"
            whole_brain_comparison_df = compare_radiomics_features(original_df, defaced_df, comparison_csv, plot_path, "Whole Brain")
            
            # Voxel intensity comparison for whole brain
            voxel_plot = f"{output_dir}/whole_brain_voxel_intensity_plot_{timestamp}.png"
            plot_voxel_intensity_comparison(original_output, defaced_output, voxel_plot, "Whole Brain")

        if amygdala_original_df is not None and amygdala_defaced_df is not None:
            # Amygdala comparison
            amygdala_comparison_csv = f"{output_dir}/amygdala_comparison_{timestamp}.csv"
            amygdala_plot_path = f"{output_dir}/amygdala_comparison_plot_{timestamp}.png"
            amygdala_comparison_df = compare_radiomics_features(amygdala_original_df, amygdala_defaced_df, 
                                                              amygdala_comparison_csv, amygdala_plot_path, "Amygdala")
            
            # Voxel intensity comparison for amygdala
            amygdala_voxel_plot = f"{output_dir}/amygdala_voxel_intensity_plot_{timestamp}.png"
            plot_voxel_intensity_comparison(original_output, defaced_output, amygdala_voxel_plot, "Amygdala")
    else:
        print("Error: Failed to run brain extractions")

if __name__ == "__main__":
    main()
