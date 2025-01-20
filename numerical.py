# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming you have a DataFrame 'df' with columns:
# 'scan_id', 'original_biomarker', 'defaced_biomarker'

# Calculate absolute differences
df['absolute_difference'] = df['original_biomarker'] - df['defaced_biomarker']

# Calculate percentage differences
df['percentage_difference'] = (df['absolute_difference'] / df['original_biomarker']) * 100

# Display first few rows
print(df.head())

# Scatter plot of Original vs Defaced biomarker values
plt.figure(figsize=(8, 6))
plt.scatter(df['original_biomarker'], df['defaced_biomarker'], alpha=0.7, c='blue')
plt.plot([df['original_biomarker'].min(), df['original_biomarker'].max()],
         [df['original_biomarker'].min(), df['original_biomarker'].max()],
         'r--', label='Line of Identity')
plt.xlabel('Original Biomarker Values')
plt.ylabel('Defaced Biomarker Values')
plt.title('Original vs Defaced Biomarker Values')
plt.legend()
plt.grid(True)
plt.show()

# Histogram of Percentage Differences
plt.figure(figsize=(8, 6))
plt.hist(df['percentage_difference'], bins=30, edgecolor='black', color='green')
plt.xlabel('Percentage Difference (%)')
plt.ylabel('Frequency')
plt.title('Distribution of Percentage Differences')
plt.grid(True)
plt.show()
