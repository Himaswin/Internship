# Brain Extraction Image Comparison Tools

This repository contains two different methods for comparing brain images before and after extraction using the Brain Extractor tool.

## Required Installations
```bash
pip3 install nibabel matplotlib numpy
```

## Method 1: Side-by-Side Comparison

**File:** `comparison.py`

This method generates side-by-side comparisons of brain slices before and after extraction in a single image.

### Features
- Creates direct side-by-side comparisons
- Generates 15 comparison slices
- Saves all comparisons in one directory
- Shows before and after images on the same figure

### Usage
```bash
python3 comparison.py
```

## Method 2: Separate Comparison

**File:** `comparison2.py`

This method generates input and output images separately before and after extraction.

### Features
- Creates input and output images
- Generates 15 comparison slices
- Saves inputs and outputs in separate directories

### Usage
```bash
python3 comparison2.py
```

