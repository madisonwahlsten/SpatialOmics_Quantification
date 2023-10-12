#  SpatialOmics_Quantification
## Overview
This script processes spatial genomics data, deriving statistical analysis and generating visualizations of specific gene expressions, and their alignments in relation to cellular structures within a given spatial region. The region is determined by cropping original images and the corresponding spot data. This processed data is then visualized in multiple ways, providing insights into the spatial relationships and density distributions of gene expressions in the tissue.

## How It Works
Image Cropping:
The script expects an image file which represents gene expression (or other cellular features) and a CSV file containing the locations (x, y) and names (name) of each identified gene expression spot.
It crops the original image and spot data to focus on an area of interest defined by coordinates (x0, x1, y0, y1).

Spline Calculation:
Spline curves are calculated based on the edges derived from the cropped image, which presumably represent cell edges or other structural features of interest.

Density Map Calculation:
For each unique gene (or gene group) identified in the spot data, a 2D Kernel Density Estimate (KDE) is computed over a grid covering the region of interest.
Gradient and Normal Vector Calculation:

Gradients and their corresponding normal vectors of the density map for each gene are computed.

Alignment Analysis:
The alignment between the gradients of the gene expression density and the calculated splines (from the cellular structures) is analyzed and stored.

Visualization:
Various visualizations are generated including overlays of spline curves, gradients, and density maps upon the cropped image, and heatmaps of the density of gene expressions.

## Prerequisites
Python 3.x
Libraries: os, skimage, pandas (pd), numpy (np), (Any additional libraries used for visualization or statistical analysis)


## Configuration
Paths
imagePath:
Path to the original image to be analyzed. If the cropped image does not exist, this image will be cropped according to the specified coordinates and used for further analysis.
figurePath:
Path where all the generated figures and intermediate results will be saved.
Example: '/path_to_your_output_directory'
dataPath:
Path to the CSV file that has been exported from Spatial Genomics. Ensure it has the required columns: 'name', 'x', and 'y'.
Example: '/path_to_your_csv_file'
## Parameters
Cropping coordinates: x0, x1, y0, y1
Other parameters to tweak based on your dataset:
Spline calculation: smoothing parameter (s)
Gradient computation
Density map: grid bounds and bin numbers
Any scaling/padding factors
Running the Script
Configure the file paths and parameters as per your dataset.
Execute the script using a Python interpreter.
