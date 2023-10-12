# %%
import os
import pickle
import time

import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev, interp1d
from statsmodels.sandbox.stats.multicomp import multipletests
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.spatial import cKDTree
from scipy.stats import ttest_1samp

import matplotlib.pyplot as plt
from matplotlib.path import Path
import seaborn as sns

import skimage
skimage.io.use_plugin('pil')

import warnings
warnings.filterwarnings('ignore')

# Where original image is stored
imagePath = ''
# Where you want all figures to be saved
figurePath = '/Users/madisonwahlsten/Documents/Graduate School/Raj_Lab/Yael_Pasadena_Run_Output/Gradients/Skimage/Crop2_FINAL' 
# CSV exported from Spatial Genomics (must have gene names as 'name', 'x', and 'y' columns)
dataPath = '/Users/madisonwahlsten/Documents/Graduate School/Raj_Lab/Yael_Pasadena_Run_Output/allTranscripts.csv'
os.makedirs(figurePath, exist_ok=True)

def main():
    
    # Coordinates used to crop image and corresponding spots exported from Spatial Genomics
    x0 = 14000
    x1 = 18000
    y0 = 16000
    y1 = 21000
    
    # If using a raw image, load using skimage and then use crop_image to achieve the following
    try:
        img = skimage.io.imread(f'{figurePath}/nucleiImage_CROP.tiff')
    except:
        img = skimage.io.imread(imagePath)
        img = crop_image(img, x0, x1, y0, y1)
    
    spots = pd.read_csv(dataPath).loc[:,['name', 'x', 'y']]
    spots = crop_spots(spots, x0, x1, y0, y1)
    print('Image and spots loaded...')
    
    # Calculate spline from the edge - use preprocess_image to form edge (d should be approximate size of cells' diameter)
    try:
        edge_dilated = skimage.io.imread(f'{figurePath}/nucleiImage_edgeDetection_d=120.png')
    except:
        edge_dilated = preprocess_image(img) # Adjust d if needed
        skimage.io.imsave(f'{figurePath}/nucleiImage_edgeDetection_d=120.png')

    spline_data = calculate_splines(edge_dilated, s=5000000) # Adjust smoothing as needed
    spline_positions = np.vstack([data[0] for data in spline_data]).reshape(-1, 2)
    spline_normals = np.vstack([data[1] for data in spline_data]).reshape(-1, 2)
    
    # Normalize the spline_normals to be unit vectors
    magnitudes = np.linalg.norm(spline_normals, axis=1)
    spline_normals_unit = spline_normals / magnitudes[:, np.newaxis]
    print('Spline calculated...')
    plot_spline_with_normals_on_image(img, spline_positions, spline_normals_unit, scale_factor=50)
    
    binNumbers = 300
    # Determine the bounds of the grid to calculate the density map from image dimensions
    x_min, x_max = spots['x'].min(), spots['x'].max()
    y_min, y_max = spots['y'].min(), spots['y'].max()
    padding = 0.05  # Extend the grid slightly beyond the min and max for better visualization
    x_min -= padding
    x_max += padding
    y_min -= padding
    y_max += padding

    x = np.linspace(x_min, x_max, binNumbers)
    y = np.linspace(y_min, y_max, binNumbers)
    X, Y = np.meshgrid(x, y)
    xy_grid = np.vstack([X.ravel(), Y.ravel()]).T
    
    try:
        density_maps = pd.read_hdf(f'{figurePath}/geneAveragedDensityMaps.h5', key='df')
    except:
        density_maps = spots.groupby('name').apply(lambda g: compute_kde(g[['x', 'y']].values, xy_grid, X)).to_frame()
        density_maps.to_hdf(f'{figurePath}/geneAveragedDensityMaps.h5', key='df')
    density_maps = density_maps.rename(columns={0:'DensityMap'})
    print('Density maps calculated...')
    # Compute gradient and normal for each density map
    density_maps['Gradient'] = density_maps['DensityMap'].apply(compute_density_gradient)
    density_maps['UnitGradient'] = density_maps['Gradient'].apply(lambda g: make_unit_vector_field(g[0], g[1]))
    density_maps['GradientNormal'] = density_maps['Gradient'].apply(lambda g: compute_normal_vectors(g[0], g[1]))
    density_maps['UnitGradientNormal'] = density_maps['GradientNormal'].apply(lambda g: make_unit_vector_field(g[0], g[1]))
    print('Density Map Gradients and Normal Vectors calculated...')
    
    # Performing analysis to determine vector alignments
    try:
        align_result = pd.read_hdf(f'{figurePath}/statisticalAnalysis_geneFlow_v2.h5', key='df')
    except:
        align_result = alignment_with_spline_normals(spline_data, density_maps, xy_grid)
        align_result.to_hdf(f'{figurePath}/Alignment_geneFlow_v2.h5', key='df')
    align_result = align_result.sort_values(by='Average Dot Product', ascending=False)
    print('Gene/spline vector alignment complete...')
    distances_result = density_maps['DensityMap'].apply(lambda g: compute_weighted_centroid(g, spline_data[0][0]))
    distances_result = pd.Series(MinMaxScaler().fit_transform(distances_result.values.reshape(1, -1)).flatten(), index=distances_result.index, name='WeightedDistance')
    distances_result.to_hdf(f'{figurePath}/WeightedDistance_geneFlow_v2.h5', key='df')
    print('Weighted centroid calculation complete...')
    analysis_result = pd.merge(align_result, distances_result, left_on='Group Name', right_index=True)
    analysis_result['WeightedScore'] = compute_score(analysis_result['Average Dot Product'].values, analysis_result['WeightedDistance'].values)
    analysis_result.to_hdf(f'{figurePath}/statisticalAnalysis_geneFlow_v2.h5', key='df')
    analysis_result = analysis_result.sort_values('Average Dot Product', ascending=False)
    print('Statistical analysis of gene/spline vector alignment complete...')
    # Plot top aligned genes
    print('Plotting gradients and normals on top of image...')
    plot_spline_aligned_genes(img, analysis_result, spline_data, density_maps, X, Y)
    plot_spline_aligned_genes(img, analysis_result, spline_data, density_maps, X, Y, top=False)
    '''
    If you have select genes you want to plot (example):
    positiveControlGenes = ['Ada', 'Apoa4', 'Apoa1', 'Aldob']
    plot_spline_aligned_genes(img, analysis_result, spline_data, density_maps, X, Y, selectGenes=positiveControlGenes)
    plot_dot_product_heatmap(analysis_result, list(analysis_result.iloc[:10,0])+positiveControlGenes+['Pmepa1','Muc2'], X, Y)
    '''
    
    # Plot density plot of indicated genes
    top10_aligned = list(analysis_result.iloc[:10,0])
    bottom10_aligned = list(analysis_result.iloc[-10:,0])
    plot_dot_product_heatmap(analysis_result, list(analysis_result.iloc[:10,0]), X, Y)

    subPath = figurePath+'/DensityPlots'
    print('Plotting gradients on top of density maps...')
    for gene in top10_aligned+bottom10_aligned:
        plot_gene_density_plot(density_maps, gene, X, Y, plotGradient=True, save=subPath)
    
def crop_image(img, x0, x1, y0, y1):
    """
    Crop the input image to the specified coordinates.

    Parameters:
        img (array-like): The input image to be cropped.
        spots (array-like): The relevant spots data (currently not utilized in the function).
        x0 (int): The starting x-coordinate for cropping.
        x1 (int): The ending x-coordinate for cropping.
        y0 (int): The starting y-coordinate for cropping.
        y1 (int): The ending y-coordinate for cropping.

    Returns:
        array-like: The cropped section of the input image.
    """
    return img[x0:x1,y0:y1]

def preprocess_image(img, d=120):
    """
    Preprocess the image by applying several filtering and morphology operations 
    to combine individual cells to identify neighborhood edges.

    Parameters:
        img (array-like): The input image to be preprocessed.
        d (int, optional): The diameter of the disk used for binary dilation in morphology. Default is 120.

    Returns:
        array-like: The processed image, which has undergone filtering and morphological operations.
    """
    img = skimage.exposure.rescale_intensity(img)
    img_smooth = skimage.img_as_ubyte(skimage.filters.gaussian(img))
    img2 = img_smooth > skimage.filters.threshold_otsu(img_smooth)
    img3 = skimage.img_as_ubyte(skimage.morphology.binary_dilation(img2, skimage.morphology.disk(d)))
    raw_edges = img3 - skimage.morphology.erosion(img3)
    edges = skimage.morphology.binary_dilation(raw_edges - skimage.morphology.erosion(raw_edges), skimage.morphology.square(5))
    return skimage.img_as_ubyte(edges)

def crop_spots(spots, x0, x1, y0, y1):
    """
    Crop the spots data based on specified x and y coordinates, updating the coordinates accordingly.

    Parameters:
        spots (DataFrame): A DataFrame containing the spots' coordinates with columns labeled 'x' and 'y'
        x0 (int): The starting x-coordinate for cropping.
        x1 (int): The ending x-coordinate for cropping.
        y0 (int): The starting y-coordinate for cropping.
        y1 (int): The ending y-coordinate for cropping.

    Returns:
        DataFrame: A DataFrame with the cropped spots and updated x and y coordinates.
    """
    temp = spots.query('((x >= @x0) and x < @x1) and ((y >= @y0) and (y < @y1))')
    temp['x'] = temp['x'] - x0
    temp['y'] = temp['y'] - y0
    return temp

def calculate_splines(edge_dilated, s=1000000, k=3, width_threshold=10, curvature_threshold=0.2):
    """
    Calculate splines for the longest contour in a binary edge image.

    Parameters:
        edge_dilated (array-like): A binary image with edges delineated.
        s_value (float): The smoothness parameter for the spline generation.

    Returns:
        list: A single-item list containing a tuple of xy coordinates and normal vectors for the spline.
    """
    def is_v_shaped(curvatures, idx):
        """Check if the curvature around an index represents a V-shape."""
        left = idx - width_threshold
        right = idx + width_threshold
        
        segment_before = curvatures[max(0, left):idx]
        segment_after = curvatures[idx + 1:min(len(curvatures), right)]
        segment_curvatures = np.concatenate([segment_before, segment_after])
        
        # Check if the segment contains mostly low-curvature values except for the middle
        v_shaped = np.all(np.abs(segment_curvatures) < curvature_threshold) and np.abs(curvatures[idx]) > curvature_threshold
        return v_shaped

    contours = skimage.measure.find_contours(edge_dilated, 0.8)
    longest_contour = max(contours, key=len)
    tck, u = splprep(np.flip(longest_contour, axis=1).T, s=s, k=k)
    u_new = np.linspace(0, 1, 1000)
    xy_new, dxy_new = np.array(splev(u_new, tck)), np.array(splev(u_new, tck, der=1))
    
    # Compute curvature
    dx, dy = dxy_new
    ddx, ddy = splev(u_new, tck, der=2)
    curvatures = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    
    # Identify V-shaped points
    v_indices = [idx for idx, curv in enumerate(curvatures) if np.abs(curv) > curvature_threshold and is_v_shaped(curvatures, idx)]
    '''
    # Visual diagnostics
    plt.figure(figsize=(10, 10))
    plt.plot(xy_new[0], xy_new[1], 'b-', label="Original Spline")
    for idx in v_indices:
        plt.plot(xy_new[0][idx], xy_new[1][idx], 'ro')  # Highlight V-shaped points in red
    plt.title("Identified V-Shaped Points")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.show()
    '''
    # Remove V-shaped points
    remove_range = 10  # How many points on either side of a V-shape to remove
    removal_indices = []
    for idx in v_indices:
        left = max(0, idx - remove_range)
        right = min(len(xy_new[0]), idx + remove_range)
        removal_indices.extend(range(left, right))
    
    # Ensure unique indices for removal
    removal_indices = list(set(removal_indices))
    

    x_modified = np.delete(xy_new[0], removal_indices)
    y_modified = np.delete(xy_new[1], removal_indices)

    # Recompute the spline with removed points
    tck, u = splprep([x_modified, y_modified], s=s, k=k)
    u_final = np.linspace(0, 1, 3000)
    xy_final, dxy_final = np.array(splev(u_final, tck)), np.array(splev(u_final, tck, der=1))
    
    normal_vectors = np.array([-dxy_final[1], dxy_final[0]]).T
    return [(xy_final.T, normal_vectors, tck)]  # Return spline data as a single-item list for consistency

def interpolate_spline_to_data(x_spline, y_spline, x_data, y_data):
    """
    Interpolate the spline to match the number of data points.

    Parameters:
        x_spline (array-like): X-coordinates of the spline points.
        y_spline (array-like): Y-coordinates of the spline points.
        x_data (array-like): X-coordinates of the data points.
        y_data (array-like): Y-coordinates of the data points.

    Returns:
        tuple: Interpolated spline points (x_interpolated, y_interpolated).
    """
    # Create interpolation functions for x and y coordinates of the spline
    f_x = interp1d(np.linspace(0, 1, len(x_spline)), x_spline, kind='linear')
    f_y = interp1d(np.linspace(0, 1, len(y_spline)), y_spline, kind='linear')

    # Interpolate the spline to match the number of data points
    x_interpolated = f_x(np.linspace(0, 1, len(x_data)))
    y_interpolated = f_y(np.linspace(0, 1, len(y_data)))

    return x_interpolated, y_interpolated

def calculate_point_distance(x_spline, y_spline, x_data, y_data):
    """
    Calculate distances from each data point to each point on the spline.

    Parameters:
        x_spline (array-like): X-coordinates of the spline points.
        y_spline (array-like): Y-coordinates of the spline points.
        x_data (array-like): X-coordinates of the data points.
        y_data (array-like): Y-coordinates of the data points.

    Returns:
        2D array: Distance from data points to spline
    """
    x_interpolated, y_interpolated = interpolate_spline_to_data(x_spline, y_spline, x_data, y_data)
    
    # Calculate the perpendicular distances to the spline
    segment_lengths = np.zeros((len(x_data), len(x_data)))
    for i, x in enumerate(x_data):
        y = y_data[i]
        dx = x_interpolated - x
        dy = y_interpolated - y
        dist = np.sqrt(dx**2 + dy**2)
        segment_lengths[:,i] = dist #Dim 0 = spline, Dim 1 = data
    return segment_lengths

def calculate_close_points(pointDistances, threshold=100):
    """
    Calculate which data points fall within (threshold) number of pixels perpendicular distance of the spline's length.

    Parameters:
        x_spline (array-like): X-coordinates of the spline points.
        y_spline (array-like): Y-coordinates of the spline points.
        spline_length (float): Length of the spline in xy-space previously calculated.
        x_data (array-like): X-coordinates of the data points.
        y_data (array-like): Y-coordinates of the data points.
        threshold (float): Percentile below which to consider data points "close" to the spline.

    Returns:
        list: Indices of close data points
        1D array: Minimum distance from data points to spline
    """
    segment_lengths = np.min(pointDistances, axis=0)
    # Find indices of close points within the threshold
    close_points_indices = np.argwhere(segment_lengths < threshold)
    return close_points_indices.flatten(), segment_lengths

def calculate_spot_metrics(data, splines, spline_lengths, threshold=100):
    """
    Calculate spot metrics for grouped data based on spots that fall within 'threshold' distance of spline.

    Parameters:
    - data (DataFrame): The input data containing spots, expected to have 'name', 'x', and 'y' columns.
    - splines (list of tuples): A list where each tuple contains the x and y coordinates of a spline.
    - spline_lengths (list): A list containing lengths of splines (not used in the function body).
    - threshold (float, optional): A threshold value for calculating close points (unit: pixels). Default is 100.

    Returns:
    - DataFrame: A DataFrame containing calculated metrics (mean, standard deviation, coefficient of variation, etc.)
      for each group ('name') and spline, in a multi-index column format.
    """
    results = []

    for name, group_data in data.groupby('name'):
        print(name)
        group_results = {'name': name}
        x_group = group_data['x'].values
        y_group = group_data['y'].values

        for idx, (spline_x, spline_y) in enumerate(splines):
            dists_all = calculate_point_distance(spline_x, spline_y, x_group, y_group)
            close_points_indices, dists = calculate_close_points(dists_all, threshold)
            print(close_points_indices.size)
            m = np.mean(dists)
            s = np.std(dists)
            cv = s / m

            group_results[f'DistanceMean_{idx}'] = m
            group_results[f'DistanceStdDeviation_{idx}'] = s
            group_results[f'CoefficientOfVariation_{idx}'] = cv
            group_results[f'Distances_{idx}'] = dists
            group_results[f'ClosePoints_{idx}'] = close_points_indices

        results.append(group_results)

    # Create a DataFrame from the results with multi-index columns
    result_df = pd.DataFrame(results)
    result_df = result_df.set_index('name')

    # Create multi-index columns
    columns = pd.MultiIndex.from_tuples([(col.split('_')[0], int(col.split('_')[1])) for col in result_df.columns], names=['Statistic', 'SplineNumber'])
    result_df.columns = columns

    return result_df.stack(level='Statistic')

def compute_kde(data, xy_grid, X):
    """
    Compute Kernel Density Estimation (KDE) on given data and grid points to form a 2D density map.

    Parameters:
    - data (array-like): Input data points to compute KDE.
    - xy_grid (array-like): The grid over which the KDE is computed.
    - X (array-like): A reference shape for reshaping the output KDE.

    Returns:
    - ndarray: KDE computed and reshaped according to input shape `X`.
    """
    kde = KernelDensity(bandwidth=100, kernel='gaussian')
    kde.fit(data)
    Z = np.exp(kde.score_samples(xy_grid))
    return Z.reshape(X.shape)

def normalize_map(density_map):
    """
    Normalize a density map to have values between 0 and 1.
    
    Parameters:
        density_map (np.array): A 2D density map.
    
    Returns:
        np.array: Normalized density map.
    """
    min_val = np.min(density_map)
    max_val = np.max(density_map)
    return (density_map - min_val) / (max_val - min_val)

def cluster_density_maps(density_maps, n_clusters=5):
    """
    Cluster density maps using KMeans and add the results directly to the dataframe.
    
    Parameters:
        density_maps (arr): Contains all density maps to be clustered
        n_clusters (int): The number of clusters for KMeans.

    Returns:
        labels (np.array): The cluster label for each density map.
        centroids (np.array): The centroids of the clusters.
    """
    
    # Use KMeans to cluster the flattened vectors
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(density_maps)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    
    return labels, centroids

def cluster_gradient_maps(gradients, n_clusters=5):
    """
    Cluster the gradients of density maps.

    Parameters:
        gradients (list of tuples): Each tuple contains two 2D arrays (gradient_x, gradient_y) for a density map.
        n_clusters (int): Number of clusters.

    Returns:
        labels (np.array): Cluster labels for each density map.
    """
    scaler = StandardScaler()
    
    # Standardize and concatenate the gradient maps
    gradient_vectors = []
    for gx, gy in gradients:
        gx_scaled = scaler.fit_transform(gx.ravel().reshape(-1, 1))
        gy_scaled = scaler.fit_transform(gy.ravel().reshape(-1, 1))
        combined = np.hstack((gx_scaled, gy_scaled))
        gradient_vectors.append(combined.ravel())
    
    gradient_matrix = np.vstack(gradient_vectors)
    
    # Cluster using KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(gradient_matrix)
    
    return kmeans.labels_, kmeans.cluster_centers_


def plot_density_clusters(centroids, map_size):
    # Visualize the centroid of each cluster as a density map
    fig, axes = plt.subplots(1,len(centroids))
    for idx, centroid in enumerate(centroids):
        plt.sca(axes.flat[idx])
        plt.imshow(centroid.reshape(map_size), cmap='viridis')
        plt.axis('off')
        plt.title(f'Cluster {idx} Centroid')
        plt.colorbar()
    plt.tight_layout()
    plt.savefig(f'{figurePath}/densityMaps_clustered_k={len(centroids)}.png', bbox_inches='tight', dpi=400)

def compute_density_gradient(density_map, threshold=0.5e-8):
    """
    Compute the gradient of a density map.

    Parameters:
        density_map (np.array): A 2D array representing the density map.
        threshold (float, optional): A threshold value below which gradient magnitudes are set to zero. Default is 0.01.

    Returns:
        gradient_x (np.array), gradient_y (np.array): Gradients of the density map along x and y axes.
    """
    gradient_y, gradient_x = np.gradient(density_map)
    # Set edge gradients to 0 to prevent edge effects from cropping
    # Left edge
    gradient_x[:, 0] = 0
    # Right edge
    gradient_x[:, -1] = 0
    # Top edge
    gradient_y[0, :] = 0
    # Bottom edge
    gradient_y[-1, :] = 0

    # Calculate the magnitude of the gradient
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # Wherever the magnitude is less than the threshold, set gradients to 0
    gradient_x[magnitude < threshold] = 0
    gradient_y[magnitude < threshold] = 0

    return gradient_x, gradient_y

def compute_normal_vectors(Vx, Vy):
    """
    Compute the normal vectors to a 2D vector field.

    Parameters:
        Vx (np.array): x-components of the original vector field.
        Vy (np.array): y-components of the original vector field.

    Returns:
        Nx (np.array), Ny (np.array): x and y components of the normal vectors.
    """
    Nx = -Vy
    Ny = Vx
    return Nx, Ny

def make_unit_vector_field(x, y):
    """
    Transform vector field into unit vectors (magnitude = 1).

    Parameters:
        x (np.array): x-component of the vector field.
        y (np.array): y-component of the vector field.

    Returns:
        unit_x (np.array), unit_y (np.array): Normalized components of the vector field.
    """
    magnitude = np.sqrt(x**2 + y**2)
    with np.errstate(divide='ignore', invalid='ignore'):  # Ignore warnings for division by zero
        unit_x = np.where(magnitude > 0, x / magnitude, 0)
        unit_y = np.where(magnitude > 0, y / magnitude, 0)
    return unit_x, unit_y

def calculate_nearest_neighbors(spline_positions, group_positions, k=1):
    """
    Find the `k` nearest spline_positions for each point in group_positions using a KDTree.

    Parameters:
    - spline_positions (ndarray): Array of reference points.
    - group_positions (ndarray): Array of points for which nearest neighbors in spline_positions need to be found.
    - k (int, optional): Number of nearest neighbors to find. Default is 1.

    Returns:
    - ndarray: Array of lists, where each list contains indices of `k` nearest neighbors in spline_positions 
      for each point in group_positions.
    """
    nearest_neighbor_indices = []
    combined_positions = np.vstack([spline_positions, group_positions])
    kdtree = cKDTree(combined_positions)
    for j in range(len(group_positions)):
        distances, indices = kdtree.query(group_positions[j], k=len(group_positions)+k+1)
        spline_indices = [w for _,w in sorted(zip(distances, indices), key=lambda r:r[0])]
        spline_indices = [idx for idx in spline_indices if idx < len(spline_positions)]
        if len(spline_indices) > k:
            spline_indices = spline_indices[:k]
        nearest_neighbor_indices.append(spline_indices)
    nearest_neighbor_indices = np.array(nearest_neighbor_indices)
    if k == 1:
        nearest_neighbor_indices = nearest_neighbor_indices.flatten()
    return nearest_neighbor_indices

def calculate_nearest_neighbors_by_projection(spline_positions, group_positions):
    # Pre-calculate spline segments and segment lengths
    starts = spline_positions[:-1]
    ends = spline_positions[1:]
    segment_vectors = ends - starts
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    normalized_segments = segment_vectors / segment_lengths[:, np.newaxis]

    # Calculate projections for all group positions onto all spline segments
    to_starts = group_positions[:, np.newaxis, :] - starts
    projections = np.sum(to_starts * normalized_segments, axis=2)

    # Ensure projections fall within the segment span
    projections = np.clip(projections, 0, segment_lengths)

    # Calculate the projected points for all group positions onto all spline segments
    projected_points = starts + projections[:, :, np.newaxis] * normalized_segments
    distances_to_projected = np.linalg.norm(group_positions[:, np.newaxis, :] - projected_points, axis=2)

    # Calculate the direct Euclidean distances to all spline points
    kdtree = cKDTree(spline_positions)
    distances_to_spline, nearest_indices = kdtree.query(group_positions, k=1)

    # Compare the two distances and choose the smaller one
    projected_indices = np.argmin(distances_to_projected, axis=1)
    use_projection = distances_to_projected[np.arange(len(group_positions)), projected_indices] < distances_to_spline
    nearest_indices[use_projection] = projected_indices[use_projection]

    return nearest_indices
    
def alignment_with_spline_normals(spline_data, density_maps, group_positions):
    """
    Computes the alignment between gene gradients and the normals of a given spline.

    Parameters:
    - spline_data (list of ndarrays): A list containing arrays of spline points and normal vectors.
    - density_maps (DataFrame): A DataFrame containing the density maps and calculated gradients. 
                                The column 'UnitGradient' should contain the unit gradients of each gene.
    - group_positions (ndarray): 2D array containing positions of the groups.

    Returns:
    - DataFrame: A DataFrame containing the results of the alignment analysis. The DataFrame includes:
        - 'Group Name': Names of the gene groups.
        - 'Average Dot Product': The average dot product value indicating the alignment between the gene gradient 
                                 and the spline's normals.
        - 'P-Value': p-values from one-sample t-test comparing the dot products to 0 (indicating orthogonality).
        - 'FDR Corrected P-Value': p-values after False Discovery Rate correction.
        - 'Reject Null': A Boolean column indicating if the null hypothesis was rejected.

    Notes:
    - This function evaluates the alignment by calculating dot products between each gene gradient and its closest 
      spline normal. A dot product close to 1 indicates good alignment, while a dot product close to 0 indicates 
      orthogonality. The alignment is statistically tested by performing a one-sample t-test for each gene, comparing 
      its set of dot products against 0.
    """
    spline_positions = np.vstack([data[0] for data in spline_data]).reshape(-1, 2)
    spline_normals = np.vstack([data[1] for data in spline_data]).reshape(-1, 2)
    
    # Normalize the spline_normals to be unit vectors
    magnitudes = np.linalg.norm(spline_normals, axis=1)
    spline_normals_unit = spline_normals / magnitudes[:, np.newaxis]
    
    # Calculate which spline vector should be used to calculate alignment of gene
    #nearest_neighbor_indices = calculate_nearest_neighbors_by_projection(spline_positions, group_positions)
    #pickle.dump(nearest_neighbor_indices, open(f'{figurePath}/nearestNeighborIndices.pkl', 'wb'))
    nearest_neighbor_indices = pickle.load(open(f'{figurePath}/nearestNeighborIndices.pkl', 'rb'))
    
    group_names = list(density_maps.index)
    avg_dot_products_per_group = []
    all_dot_products = []
    p_values = []
    
    # Compute dot products between each group vector and the closest spline vector
    for group_gradient in density_maps['UnitGradient'].values:
        gradient_x, gradient_y = group_gradient
        flattened_group_gradient = np.vstack([gradient_x.ravel(), gradient_y.ravel()]).T
        # Compute magnitudes for each group vector
        gradient_magnitudes = np.linalg.norm(flattened_group_gradient, axis=1)
        
        # Compute dot products for each group vector with the closest spline vector
        dot_products = np.abs(np.array([
        np.dot(spline_normals_unit[nearest_neighbor_indices[j]], flattened_group_gradient[j])
        for j in range(len(flattened_group_gradient))]))
        dot_products_nonZero = np.abs(np.array([
        np.dot(spline_normals_unit[nearest_neighbor_indices[j]], flattened_group_gradient[j])
        for j in range(len(flattened_group_gradient)) if gradient_magnitudes[j] > 0]))
        all_dot_products.append(dot_products)
        avg_dot_products_per_group.append(np.mean(dot_products_nonZero))
        
        # Perform one-sample t-test
        t_stat, p_value = ttest_1samp(dot_products, 0)  # Testing against 0 since dot product of orthogonal vectors is 0
        p_values.append(p_value)
    
    # Perform FDR correction
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
    
    # Store results in a DataFrame
    result_df = pd.DataFrame({
        'Group Name': group_names,
        'Dot Products': all_dot_products,
        'Average Dot Product': avg_dot_products_per_group,
        'P-Value': p_values,
        'FDR Corrected P-Value': pvals_corrected,
        'Reject Null': reject
    })
    
    return result_df

def compute_weighted_centroid(density_map, spline_positions):
    """
    Compute the weighted centroid of a density map and the minimum distance from the centroid to any point on the spline.

    Parameters:
        density_map (np.array): A 2D array representing the density map.
        spline_positions (np.array): Nx2 array of xy-coordinates for the spline.

    Returns:
        centroid_to_spline_distance (float): The minimum distance from the centroid to any point on the spline.
    """
    y_indices, x_indices = np.indices(density_map.shape)
    x_centroid = np.sum(x_indices * density_map) / np.sum(density_map)
    y_centroid = np.sum(y_indices * density_map) / np.sum(density_map)
    
    centroid = np.array([y_centroid, x_centroid])
    
    # Calculate distances from centroid to each point on spline
    distances = np.linalg.norm(spline_positions - centroid, axis=1)

    # Get the minimum distance
    centroid_to_spline_distance = np.min(distances)

    return centroid_to_spline_distance

def compute_score(alignment_scores, centroid_distances, alpha=0.5):
    """
    Compute a combined score for each gene that evaluates the alignment of its gradient with spline normals
    and its weighted centroid distance.
    
    Parameters:
    - alignment_scores (ndarray or list): Average dot product values representing alignment scores for each gene.
    - centroid_distances (ndarray or list): Weighted centroid distances for each gene.
    - alpha (float): Weighting factor between [0, 1] to balance alignment and centroid distance. 
                     Default is 0.5.
    
    Returns:
    - ndarray: Combined scores for each gene, representing alignment and weighted centroid distance.
    """
    
    # Element-wise combination of the scores for each gene
    combined_scores = alpha * np.array(alignment_scores) + (1 - alpha) * np.array(centroid_distances)

    return combined_scores

def density_in_boxes(spline_data, density_maps, bin_size, height_range):
    """
    Calculate the sum of density values in boxes defined along splines and specified heights.
    **** UNTESTED ****

    Parameters:
    - spline_data (tuple of ndarrays): Tuple containing arrays of spline points and normal vectors.
    - density_maps (DataFrame): A DataFrame containing density maps.
    - bin_size (int or float): Width of the box.
    - height_range (range or list): A range or list of heights to calculate density.

    Returns:
    - DataFrame: A DataFrame with multi-index (Group, Height) containing summed density values for each box.
    """
    xy_spline, normal_vectors = spline_data[0]
    results = {}  # Dictionary to store the density values

    for i, (point, normal) in enumerate(zip(xy_spline, normal_vectors)):
        for h in height_range:
            # Define the coordinates of the box
            half_width = bin_size / 2
            half_height = h / 2
            p1 = point + normal * half_height - normal_vectors[i] * half_width  # Top left point of the box
            p2 = point - normal * half_height - normal_vectors[i] * half_width  # Bottom left point of the box
            p3 = point - normal * half_height + normal_vectors[i] * half_width  # Bottom right point of the box
            p4 = point + normal * half_height + normal_vectors[i] * half_width  # Top right point of the box

            box_coords = np.array([p1, p2, p3, p4])

            # Create a mask to check if each point in the density map is within the box
            mask = np.zeros(density_maps.iloc[0]['DensityMap'].shape, dtype=bool)

            # Check if each point in the density map is within the box
            path = Path(box_coords)
            for y in range(mask.shape[0]):
                for x in range(mask.shape[1]):
                    point = np.array([x, y])
                    mask[y, x] = path.contains_point(point)
            
            # Iterate over each density map
            for j, (_, row) in enumerate(density_maps.iterrows()):
                density_map = row['DensityMap']
                
                # Sum the density values within the box
                density_sum = np.sum(density_map[mask])
                
                # Store the density sum in the results dictionary
                key = (j, h)  # Use a tuple of (group index, height) as the key
                if key not in results:
                    results[key] = 0
                results[key] += density_sum

    # Convert the results dictionary to a DataFrame for easier analysis
    result_df = pd.DataFrame.from_dict(results, orient='index', columns=['Density Sum'])
    result_df.index = pd.MultiIndex.from_tuples(result_df.index, names=('Group', 'Height'))

    return result_df

def plot_gene_density_plot(density_maps, geneName, X, Y, plotGradient=False, save=figurePath):
    """
    Plot 2D density plot generated by KDE and gene vectors on an image for visual analysis.

    Parameters:
    - density_maps (DataFrame): A DataFrame containing density maps and gradient vectors.
    - geneName (str): name of gene to plot
    - X, Y (ndarray): Meshgrid arrays.
    - plotGradient (bool, optional): Whether or not to plot gradient on top of density map.
    Returns:
    - None: The function will display and save plots but does not return any values.
    """
    os.makedirs(save, exist_ok=True)
    
    density_map = density_maps.loc[geneName, 'DensityMap']
    plt.figure(figsize=(4.8, 6.4))
    plt.title(f'Gradient Map for {geneName}')
    plt.pcolormesh(X, Y, density_map, shading='auto')
    if plotGradient:
        gradient_x, gradient_y = density_maps.loc[geneName, 'Gradient']
        gradient_y = -gradient_y # Inverting y axis does not flip quiver arrow directions for some reason, so must manually flip
        plt.quiver(X[::5, ::5], Y[::5, ::5], gradient_x[::5, ::5], gradient_y[::5, ::5], color='red')
    plt.axis('off')
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.savefig(f'{save}/KDEPlot_{geneName}.png', bbox_inches='tight', dpi=400)

def plot_spline_aligned_genes(img, analysis_result, spline_data, density_maps, X, Y, top=True, numGenes=5, selectGenes=None):
    """
    Plot splines and gene vectors on an image for visual analysis.

    Parameters:
    - img (array-like): The image to plot vectors and splines on.
    - analysis_result (DataFrame): DataFrame containing statistical analysis results.
    - spline_data (tuple of ndarrays): Tuple containing arrays of spline points and normal vectors.
    - density_maps (DataFrame): A DataFrame containing density maps and gradient vectors.
    - X, Y (ndarray): Meshgrid arrays.
    - top (bool, optional): If True, use top results from analysis_result; otherwise use bottom. Default is True.
    - numGenes (int, optional): Number of genes to visualize. Default is 5.

    Returns:
    - None: The function will display and save plots but does not return any values.
    """
    colors = ['red', 'darkorange', 'limegreen', 'deepskyblue', 'blueviolet']
    if numGenes > 5:
        print('Sorry! Only up to 5 colors available to prevent graph crowding. Remaining genes will be ignored.')
        numGenes = 5
    if selectGenes:
        top_results = selectGenes
        temp = '_'.join(top_results)+'_'
    elif top:
        top_results = analysis_result.iloc[:numGenes,0]
        temp = 'Top'
    else:
        top_results = analysis_result.iloc[-numGenes:,0]
        temp = 'Bottom'

    x_new, y_new = spline_data[0][0][:,0], spline_data[0][0][:,1]
    nx_new, ny_new = spline_data[0][1][:,0], spline_data[0][1][:,1]
    plt.figure()
    plt.imshow(img, cmap='gray')
    for i, gene in enumerate(top_results):
        density_map = density_maps.loc[gene, 'DensityMap']
        gradient_x, gradient_y = density_maps.loc[gene, 'Gradient']
        gradient_y = -gradient_y
        plt.quiver(X[::5, ::5], Y[::5, ::5], gradient_x[::5, ::5], gradient_y[::5, ::5], color=colors[i], label=gene)
    plt.plot(x_new, y_new, 'y-', lw=1)
    plt.axis('off')
    plt.legend(bbox_to_anchor=(1.04,0.7), frameon=False, title='Gene')
    plt.savefig(f'{figurePath}/GradientPlot_{temp}{numGenes}normalGenes.png', dpi=400, bbox_inches='tight')

    plt.figure()
    plt.imshow(img, cmap='gray')
    for i, gene in enumerate(top_results):
        density_map = density_maps.loc[gene, 'DensityMap']
        normal_x, normal_y = density_maps.loc[gene, 'GradientNormal']
        normal_y = -normal_y
        plt.quiver(X[::5, ::5], Y[::5, ::5], normal_x[::5, ::5], normal_y[::5, ::5], color=colors[i], label=gene)
        plt.plot(x_new, y_new, 'b-', lw=1)
    plt.axis('off')
    plt.legend(bbox_to_anchor=(1.04,0.7), frameon=False, title='Gene')
    plt.savefig(f'{figurePath}/GradientNormalPlot_{temp}{numGenes}normalGenes.png', dpi=400, bbox_inches='tight')

def plot_spline_with_normals_on_image(img, spline_positions, spline_normals, scale_factor=5, n=50):
    """
    Plots the spline and its unit normal vectors on top of an image.
    
    Parameters:
        img (np.array): Image array on which the plot is overlayed.
        spline_positions (np.array): Nx2 array of xy-coordinates for the spline.
        spline_normals (np.array): Nx2 array of unit normal vectors for each point on the spline.
        scale_factor (float): Scaling factor for visualizing the normal vectors. Defaults to 1.
    """
    # Extract x and y coordinates of spline
    x_coords, y_coords = spline_positions[:, 0], spline_positions[:, 1]

    # Calculate endpoints of the normal vectors to be plotted
    normal_endpoints = spline_positions + scale_factor * spline_normals

    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(img, cmap='gray')
    ax.plot(x_coords, y_coords, color='red')
    
    # Plot the normal vectors
    for start, normal in zip(spline_positions[::n], spline_normals[::n]):
        ax.arrow(start[0], start[1], normal[0]*50, normal[1]*50, head_width=15, head_length=20, fc='red', ec='red')
    ax.axis('off')
    fig.savefig(f'{figurePath}/splineImage_withNormals.png', dpi=400, bbox_inches='tight')

def plot_dot_product_heatmap(analysis_result, genes, X, Y):
    analysis_result = analysis_result.set_index('Group Name')
    for gene in genes:
        dotProds = analysis_result.loc[gene, 'Dot Products'].reshape(X.shape)
        plt.figure(figsize=(4.8, 6.4))
        plt.pcolormesh(X, Y, dotProds, shading='auto')
        plt.axis('off')
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.title(f'Gene = {gene}')
        plt.savefig(f'{figurePath}/gradientDotProduct_heatmap_{gene}.png', bbox_inches='tight', dpi=400)

if __name__ == "__main__":
    main()
