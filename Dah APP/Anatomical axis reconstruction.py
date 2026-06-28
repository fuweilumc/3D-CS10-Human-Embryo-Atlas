# -*- coding: utf-8 -*-
"""
Curvilinear coordinate reconstruction tool for tubular tissues V7.0 - segment-wise control with continuous coordinate mapping

V7.0 updates:
✅ Removed global rotation and plane control - now fully segment-controlled
✅ Each segment can specify a start and end point (range of midpoints)
✅ Ensures continuity of coordinate mapping - smooth transition between segments
✅ Removed Z-coordinate-based filtering - simplified interface
✅ Optimized interpolation - fixed parameters within a segment, smooth interpolation between segments

Main features:
- Segment-wise coordinate frames: different rotation angles and radii per segment
- Continuous coordinate mapping: improved interpolation ensures coordinate continuity
- Flexible segment definition: explicitly specify the start and end of each segment
"""
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
import dash
from dash import dcc, html, Input, Output, State, callback_context, ALL
import json
import base64
import pandas as pd
import scanpy as sc

# ======================
# 1. Load h5ad data
# ======================
mtx = sc.read_h5ad("whole_embryo_hvg_res_sub.py")
mtx.obs['whole_leiden_str'] = mtx.obs['leiden_hvg_sub'].astype(str)
mtx.obs['sub_leiden_str'] = mtx.obs['leiden_hvg_sub'].astype(str)

for col in ['x_centroid', 'y_centroid', 'z_centroid']:
    if col not in mtx.obs.columns:
        mtx.obs[col] = 0

cluster_labels = sorted(mtx.obs['leiden_hvg_sub'].astype(str).unique())
e_clusters = sorted(cluster_labels, key=lambda x: int(x) if x.isdigit() else x)
p_clusters = []


section_labels = sorted(mtx.obs['adjusted_cure_clustering'].unique())

cluster_colors_list = mtx.uns.get('leiden_hvg_1_colors', 
                                  px.colors.qualitative.Plotly * (len(cluster_labels) // 10 + 1))
cluster_colors = {str(c): cluster_colors_list[i] for i, c in enumerate(cluster_labels)}

parent_to_sub = {}
for parent in e_clusters:
    subs = (
        mtx.obs.loc[mtx.obs['whole_leiden_str'] == parent, 'leiden_hvg_sub']
        .replace("", np.nan).dropna().unique()
    )
    parent_to_sub[parent] = sorted(map(str, subs)) if len(subs) else []

parent_map = {sub: parent for parent, subs in parent_to_sub.items() for sub in subs}
subcluster_colors = {sub: cluster_colors.get(parent_map[sub], '#808080') for sub in parent_map}

# Global data store
test_data = {
    'cells': [],  # to be filled in callbacks
    'mtx': mtx,
    'cluster_info': {
        'e_clusters': e_clusters,
        'p_clusters': p_clusters,
        'parent_to_sub': parent_to_sub,
        'cluster_colors': cluster_colors,
        'subcluster_colors': subcluster_colors,
        'section_labels': section_labels
    }
}

# ======================
# 2. Curve and coordinate-frame functions
# ======================
def fit_smooth_curve(points, n_samples=100, smoothing=0.0):
    """
    Fit a smooth curve
    
    Parameters:
        points: array of control points
        n_samples: number of sampled points
        smoothing: smoothing parameter (0 = interpolation, >0 = smoothing fit)
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 2: 
        return points
    
    # If too few points, use linear interpolation
    if len(points) < 4:
        t_in = np.linspace(0, 1, len(points))
        t_out = np.linspace(0, 1, n_samples)
        return np.column_stack([np.interp(t_out, t_in, points[:, i]) for i in range(3)])
    
    # Try spline interpolation; if it fails, progressively increase smoothing
    smoothing_values = [smoothing]
    
    # If the initial smoothing is too small, add some fallback values
    if smoothing < 0.5:
        smoothing_values.extend([0.5, 1.0, 2.0, 5.0])
    elif smoothing < 2.0:
        smoothing_values.extend([smoothing * 1.5, smoothing * 2, smoothing * 3])
    
    for s_value in smoothing_values:
        try:
            # Increase max iterations, use more suitable parameters
            tck, u = splprep(
                [points[:, 0], points[:, 1], points[:, 2]], 
                s=s_value, 
                k=min(3, len(points)-1)
            )
            x, y, z = splev(np.linspace(0, 1, n_samples), tck)
            
            # If successful but a different smoothing was used, print a note
            if s_value != smoothing:
                print(f"⚠️ Smoothing auto-adjusted: {smoothing:.1f} → {s_value:.1f}")
            
            return np.column_stack([x, y, z])
        except Exception as e:
            if s_value == smoothing_values[-1]:
                # All attempts failed; fall back to linear interpolation
                print(f"⚠️ Spline interpolation failed; using linear interpolation. Error: {e}")
                t_in = np.linspace(0, 1, len(points))
                t_out = np.linspace(0, 1, n_samples)
                return np.column_stack([np.interp(t_out, t_in, points[:, i]) for i in range(3)])
            # Otherwise continue trying the next smoothing value
            continue

def compute_rotation_minimizing_frames(centerline):
    centerline, n = np.asarray(centerline, dtype=float), len(np.asarray(centerline))
    tangents, normals, binormals = np.zeros_like(centerline), np.zeros_like(centerline), np.zeros_like(centerline)
    tangents[0], tangents[-1], tangents[1:-1] = centerline[1] - centerline[0], centerline[-1] - centerline[-2], centerline[2:] - centerline[:-2]
    for i in range(n):
        norm = np.linalg.norm(tangents[i])
        if norm > 1e-10: tangents[i] /= norm
    
    t0, ref = tangents[0], np.array([0, 0, 1]) if abs(tangents[0][2]) < 0.9 else np.array([1, 0, 0])
    normals[0] = (ref - np.dot(ref, t0) * t0) / (np.linalg.norm(ref - np.dot(ref, t0) * t0) + 1e-10)
    binormals[0] = np.cross(t0, normals[0])
    
    for i in range(1, n):
        v1, c1 = centerline[i] - centerline[i-1], np.dot(centerline[i] - centerline[i-1], centerline[i] - centerline[i-1])
        if c1 < 1e-10: normals[i], binormals[i] = normals[i-1], binormals[i-1]; continue
        r_L = normals[i-1] - (2.0 / c1) * np.dot(v1, normals[i-1]) * v1
        t_sum, c2 = tangents[i] + tangents[i-1], np.dot(tangents[i] + tangents[i-1], tangents[i] + tangents[i-1])
        normals[i] = r_L if c2 < 1e-10 else r_L - (2.0 / c2) * np.dot(t_sum, r_L) * t_sum
        normals[i] /= (np.linalg.norm(normals[i]) + 1e-10)
        binormals[i] = np.cross(tangents[i], normals[i]) / (np.linalg.norm(np.cross(tangents[i], normals[i])) + 1e-10)
    
    return tangents, normals, binormals

def create_cylinder_with_rmf(centerline, radius=1.0, radial_segments=20):
    centerline, n_curve_points = np.asarray(centerline, dtype=float), len(np.asarray(centerline))
    tangents, normals, binormals = compute_rotation_minimizing_frames(centerline)
    vertices = []
    for i in range(n_curve_points):
        for j in range(radial_segments):
            angle = 2 * np.pi * j / radial_segments
            offset = radius * (np.cos(angle) * normals[i] + np.sin(angle) * binormals[i])
            vertices.append(centerline[i] + offset)
    vertices = np.array(vertices)
    faces = []
    for i in range(n_curve_points - 1):
        for j in range(radial_segments):
            v1, v2 = i * radial_segments + j, i * radial_segments + (j + 1) % radial_segments
            v3, v4 = (i + 1) * radial_segments + j, (i + 1) * radial_segments + (j + 1) % radial_segments
            faces.extend([[v1, v2, v4], [v1, v4, v3]])
    return vertices, np.array(faces)

def create_local_coordinate_system(center_point, centerline, radius, rotation_angle=0):
    """Create a local coordinate frame at the first midpoint"""
    center_point = np.array(center_point)
    
    if len(centerline) >= 2:
        tangent = np.array(centerline[1]) - center_point
        tangent = tangent / (np.linalg.norm(tangent) + 1e-10)
    else:
        tangent = np.array([0, 0, 1])
    
    if abs(tangent[2]) < 0.9:
        perp1 = np.array([0, 0, 1])
    else:
        perp1 = np.array([1, 0, 0])
    perp1 = perp1 - np.dot(perp1, tangent) * tangent
    perp1 = perp1 / (np.linalg.norm(perp1) + 1e-10)
    perp2 = np.cross(tangent, perp1)
    perp2 = perp2 / (np.linalg.norm(perp2) + 1e-10)
    
    angle_rad = np.radians(rotation_angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    x_axis = cos_a * perp1 + sin_a * perp2
    y_axis = -sin_a * perp1 + cos_a * perp2
    z_axis = tangent
    
    extension = radius * 1.5
    
    cross_x_pos = center_point + extension * x_axis
    cross_x_neg = center_point - extension * x_axis
    cross_y_pos = center_point + extension * y_axis
    cross_y_neg = center_point - extension * y_axis
    
    circle_points = []
    n_circle = 50
    for i in range(n_circle + 1):
        angle = 2 * np.pi * i / n_circle
        point = center_point + radius * (np.cos(angle) * x_axis + np.sin(angle) * y_axis)
        circle_points.append(point)
    
    return {
        'center': center_point,
        'x_axis': x_axis,
        'y_axis': y_axis,
        'z_axis': z_axis,
        'cross_x': [cross_x_neg, center_point, cross_x_pos],
        'cross_y': [cross_y_neg, center_point, cross_y_pos],
        'circle': np.array(circle_points)
    }

def interpolate_segment_params(position_idx, n_total_points, segments):
    """
    Interpolate the angle and radius at the current position from the position index
    
    Parameters:
        position_idx: index of the current position (0 to n_total_points-1)
        n_total_points: total number of midpoints
        segments: list of segment parameters [{'start_idx': int, 'end_idx': int, 'angle': float, 'radius': float}, ...]
    
    Returns:
        (angle, radius): interpolated angle and radius
    """
    if not segments or len(segments) == 0:
        return 0.0, 2.0
    
    # If there is only one segment, return directly
    if len(segments) == 1:
        return segments[0]['angle'], segments[0]['radius']
    
    # Ensure segments are sorted by start_idx
    sorted_segments = sorted(segments, key=lambda x: x['start_idx'])
    
    # Find the segment containing the current position
    for seg in sorted_segments:
        start = seg['start_idx']
        end = seg.get('end_idx', n_total_points - 1)  # default to the last
        
        if start <= position_idx <= end:
            # Within the current segment, use its fixed parameters without interpolation
            return seg['angle'], seg['radius']
    
    # If the position is not in any segment, find the nearest one
    # This usually happens in the gap between two segments
    for i in range(len(sorted_segments) - 1):
        seg1 = sorted_segments[i]
        seg2 = sorted_segments[i + 1]
        end1 = seg1.get('end_idx', n_total_points - 1)
        start2 = seg2['start_idx']
        
        if end1 < position_idx < start2:
            # Interpolate between two segments
            gap_length = start2 - end1
            if gap_length > 1:
                t = (position_idx - end1) / gap_length
                angle = seg1['angle'] + t * (seg2['angle'] - seg1['angle'])
                radius = seg1['radius'] + t * (seg2['radius'] - seg1['radius'])
                return angle, radius
    
    # If before the first segment, return the first segment's parameters
    if position_idx < sorted_segments[0]['start_idx']:
        return sorted_segments[0]['angle'], sorted_segments[0]['radius']
    
    # If after the last segment, return the last segment's parameters
    return sorted_segments[-1]['angle'], sorted_segments[-1]['radius']

def compute_local_frames_along_curve(centerline, rotation_angle=0):
    """
    Compute the local coordinate frame at each point along the curve (using rotation-minimizing frames)
    Detect tube flipping and automatically adjust the coordinate frame
    """
    centerline = np.array(centerline)
    n_points = len(centerline)
    
    # Compute tangent vectors
    tangents = np.zeros_like(centerline)
    tangents[0] = centerline[1] - centerline[0]
    tangents[-1] = centerline[-1] - centerline[-2]
    tangents[1:-1] = centerline[2:] - centerline[:-2]
    
    for i in range(n_points):
        norm = np.linalg.norm(tangents[i])
        if norm > 1e-10:
            tangents[i] /= norm
    
    # Compute normal and binormal vectors (rotation-minimizing frames)
    normals = np.zeros_like(centerline)
    binormals = np.zeros_like(centerline)
    
    # First point: establish the initial frame
    t0 = tangents[0]
    if abs(t0[2]) < 0.9:
        ref = np.array([0, 0, 1])
    else:
        ref = np.array([1, 0, 0])
    
    ref = ref - np.dot(ref, t0) * t0
    ref = ref / (np.linalg.norm(ref) + 1e-10)
    
    # Compute the second perpendicular vector
    perp2 = np.cross(t0, ref)
    perp2 = perp2 / (np.linalg.norm(perp2) + 1e-10)
    
    # Apply the rotation angle
    angle_rad = np.radians(rotation_angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    # normals correspond to x_axis, binormals to y_axis
    normals[0] = cos_a * ref + sin_a * perp2
    binormals[0] = -sin_a * ref + cos_a * perp2
    
    # Record the world direction of the initial Y axis
    initial_y_direction = binormals[0].copy()
    
    # Propagate the frame (using RMF)
    for i in range(1, n_points):
        v1 = centerline[i] - centerline[i-1]
        c1 = np.dot(v1, v1)
        
        if c1 < 1e-10:
            normals[i] = normals[i-1]
            binormals[i] = binormals[i-1]
            continue
        
        # Rotate the normal vector
        r_L = normals[i-1] - (2.0 / c1) * np.dot(v1, normals[i-1]) * v1
        t_sum = tangents[i] + tangents[i-1]
        c2 = np.dot(t_sum, t_sum)
        
        if c2 < 1e-10:
            normals[i] = r_L
        else:
            normals[i] = r_L - (2.0 / c2) * np.dot(t_sum, r_L) * t_sum
        
        normals[i] = normals[i] / (np.linalg.norm(normals[i]) + 1e-10)
        
        # Binormal vector
        binormals[i] = np.cross(tangents[i], normals[i])
        binormals[i] = binormals[i] / (np.linalg.norm(binormals[i]) + 1e-10)
        
        # Key: detect whether the Y axis has flipped (relative to the initial direction)
        # If the dot product of the Y axis with the initial direction is negative, it has flipped
        if np.dot(binormals[i], initial_y_direction) < 0:
            # Flip the Y axis (binormal)
            binormals[i] = -binormals[i]
            # Also flip the X axis to keep a right-handed system
            normals[i] = -normals[i]
    
    return {
        'centers': centerline,
        'tangents': tangents,
        'normals': normals,
        'binormals': binormals
    }

def compute_local_frames_along_curve_segmented(centerline, center_points, segments):
    """
    Compute the local frame at each point along the curve - supports segment parameters (using absolute angles)
    
    Key improvements:
    - Each segment uses its absolute rotation angle rather than a cumulative increment
    - First compute the base RMF frame, then apply each point's segment-specific absolute rotation
    
    Parameters:
        centerline: sampled points along the curve
        center_points: list of original midpoints
        segments: segment parameters [{'start_idx': int, 'angle': float, 'radius': float}, ...]
    
    Returns:
        a dict containing centers, tangents, normals, binormals
    """
    centerline = np.array(centerline)
    n_curve_points = len(centerline)
    n_center_points = len(center_points)
    
    # Step 1: compute tangent vectors
    tangents = np.zeros_like(centerline)
    tangents[0] = centerline[1] - centerline[0]
    tangents[-1] = centerline[-1] - centerline[-2]
    tangents[1:-1] = centerline[2:] - centerline[:-2]
    
    for i in range(n_curve_points):
        norm = np.linalg.norm(tangents[i])
        if norm > 1e-10:
            tangents[i] /= norm
    
    # Step 2: compute the base RMF frame (no rotation)
    base_normals = np.zeros_like(centerline)
    base_binormals = np.zeros_like(centerline)
    
    # Initialize the base frame at the first point (no rotation)
    t0 = tangents[0]
    if abs(t0[2]) < 0.9:
        ref = np.array([0, 0, 1])
    else:
        ref = np.array([1, 0, 0])
    
    ref = ref - np.dot(ref, t0) * t0
    ref = ref / (np.linalg.norm(ref) + 1e-10)
    
    base_normals[0] = ref
    base_binormals[0] = np.cross(t0, ref)
    base_binormals[0] = base_binormals[0] / (np.linalg.norm(base_binormals[0]) + 1e-10)
    
    initial_y_direction = base_binormals[0].copy()
    
    # Propagate the base frame using RMF
    for i in range(1, n_curve_points):
        v1 = centerline[i] - centerline[i-1]
        c1 = np.dot(v1, v1)
        
        if c1 < 1e-10:
            base_normals[i] = base_normals[i-1]
            base_binormals[i] = base_binormals[i-1]
            continue
        
        # Rotate the normal vector
        r_L = base_normals[i-1] - (2.0 / c1) * np.dot(v1, base_normals[i-1]) * v1
        t_sum = tangents[i] + tangents[i-1]
        c2 = np.dot(t_sum, t_sum)
        
        if c2 < 1e-10:
            base_normals[i] = r_L
        else:
            base_normals[i] = r_L - (2.0 / c2) * np.dot(t_sum, r_L) * t_sum
        
        base_normals[i] = base_normals[i] / (np.linalg.norm(base_normals[i]) + 1e-10)
        
        # Binormal vector
        base_binormals[i] = np.cross(tangents[i], base_normals[i])
        base_binormals[i] = base_binormals[i] / (np.linalg.norm(base_binormals[i]) + 1e-10)
        
        # Detect flipping
        if np.dot(base_binormals[i], initial_y_direction) < 0:
            base_binormals[i] = -base_binormals[i]
            base_normals[i] = -base_normals[i]
    
    # Step 3: apply each point's segment-specific absolute rotation to the base frame
    normals = np.zeros_like(centerline)
    binormals = np.zeros_like(centerline)
    
    for i in range(n_curve_points):
        # Compute the midpoint index corresponding to the current curve point
        t_param = i / (n_curve_points - 1) if n_curve_points > 1 else 0
        center_idx = int(t_param * (n_center_points - 1))
        
        # Get the absolute rotation angle at the current position
        absolute_angle, _ = interpolate_segment_params(center_idx, n_center_points, segments)
        
        # Apply the absolute rotation to the base frame
        angle_rad = np.radians(absolute_angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        
        # Rotate within the base frame
        normals[i] = cos_a * base_normals[i] + sin_a * base_binormals[i]
        binormals[i] = -sin_a * base_normals[i] + cos_a * base_binormals[i]
    
    return {
        'centers': centerline,
        'tangents': tangents,
        'normals': normals,
        'binormals': binormals
    }

def transform_to_local_coords_nearest_plane(points, local_frames):
    """
    Project points onto the nearest local plane, then compute new coordinates
    """
    points = np.array(points)
    centers = np.array(local_frames['centers'])     # NumPy array
    tangents = np.array(local_frames['tangents'])   # NumPy array
    normals = np.array(local_frames['normals'])     # NumPy array
    binormals = np.array(local_frames['binormals']) # NumPy array
    
    # Use a KD-tree to find the nearest centerline point for each point
    tree = cKDTree(centers)
    distances, indices = tree.query(points)
    
    # Compute new coordinates for each point
    new_coords = np.zeros_like(points)
    
    for i, (point, idx) in enumerate(zip(points, indices)):
        # Get the local frame of the nearest point
        center = centers[idx]
        normal = normals[idx]
        binormal = binormals[idx]
        tangent = tangents[idx]
        
        # Compute the relative vector
        relative = point - center
        
        # Project onto the local frame
        new_x = np.dot(relative, normal)     # X axis (radial 1)
        new_y = np.dot(relative, binormal)   # Y axis (radial 2)
        new_z = np.dot(relative, tangent)    # Z axis (along the curve)
        
        # Add the cumulative curve length as the Z coordinate
        if idx > 0:
            # Compute the curve length from the start to the current position
            curve_length = np.sum(np.linalg.norm(centers[1:idx+1] - centers[:idx], axis=1))
            new_z += curve_length
        
        new_coords[i] = [new_x, new_y, new_z]
    
    return new_coords, indices

def create_perpendicular_planes(local_system, centerline, radius, plane_height=None):
    """
    Create two perpendicular planes extending along the curve
    
    Parameters:
        local_system: local coordinate frame
        centerline: the centerline
        radius: cylinder radius (used for the default plane width)
        plane_height: plane height (if None, use radius * 2)
    """
    centerline = np.array(centerline)
    x_axis = np.array(local_system['x_axis'])
    y_axis = np.array(local_system['y_axis'])
    
    # If no plane height is specified, use the default (twice the radius)
    plane_width = plane_height if plane_height is not None else radius * 2
    
    n_points = len(centerline)
    tangents = np.zeros_like(centerline)
    
    tangents[0] = centerline[1] - centerline[0]
    tangents[-1] = centerline[-1] - centerline[-2]
    tangents[1:-1] = centerline[2:] - centerline[:-2]
    
    for i in range(n_points):
        norm = np.linalg.norm(tangents[i])
        if norm > 1e-10:
            tangents[i] /= norm
    
    local_x_axes = np.zeros_like(centerline)
    local_y_axes = np.zeros_like(centerline)
    
    local_x_axes[0] = x_axis
    local_y_axes[0] = y_axis
    
    for i in range(1, n_points):
        v1 = centerline[i] - centerline[i-1]
        c1 = np.dot(v1, v1)
        
        if c1 < 1e-10:
            local_x_axes[i] = local_x_axes[i-1]
            local_y_axes[i] = local_y_axes[i-1]
            continue
        
        r_L_x = local_x_axes[i-1] - (2.0 / c1) * np.dot(v1, local_x_axes[i-1]) * v1
        t_sum = tangents[i] + tangents[i-1]
        c2 = np.dot(t_sum, t_sum)
        
        if c2 < 1e-10:
            local_x_axes[i] = r_L_x
        else:
            local_x_axes[i] = r_L_x - (2.0 / c2) * np.dot(t_sum, r_L_x) * t_sum
        
        local_x_axes[i] /= (np.linalg.norm(local_x_axes[i]) + 1e-10)
        local_y_axes[i] = np.cross(tangents[i], local_x_axes[i])
        local_y_axes[i] /= (np.linalg.norm(local_y_axes[i]) + 1e-10)
    
    xz_plane_left = []
    xz_plane_right = []
    
    for i in range(n_points):
        xz_plane_left.append(centerline[i] - plane_width * local_x_axes[i])
        xz_plane_right.append(centerline[i] + plane_width * local_x_axes[i])
    
    yz_plane_left = []
    yz_plane_right = []
    
    for i in range(n_points):
        yz_plane_left.append(centerline[i] - plane_width * local_y_axes[i])
        yz_plane_right.append(centerline[i] + plane_width * local_y_axes[i])
    
    return {
        'xz_plane_left': np.array(xz_plane_left),
        'xz_plane_right': np.array(xz_plane_right),
        'yz_plane_left': np.array(yz_plane_left),
        'yz_plane_right': np.array(yz_plane_right)
    }

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Curvilinear coordinate reconstruction of the gut tube", style={'textAlign': 'center'}),
    html.Div([
        html.Div([
            html.H3("Control panel"),
            
            # Cluster and Section selection
            html.Div([
                html.Label("📊 Data filtering:", style={'fontWeight': 'bold', 'marginBottom': '10px', 'fontSize': '14px'}),
                html.Label("Select clusters:", style={'fontSize': '12px', 'marginTop': '5px'}),
                dcc.Dropdown(
                    id='cluster-selector',
                    options=[{'label': c, 'value': c} for c in e_clusters + p_clusters],
                    value=[e_clusters[0]] if e_clusters else [],
                    multi=True,
                    style={'marginBottom': '10px'}
                ),
                html.Label("Select sections:", style={'fontSize': '12px'}),
                dcc.Dropdown(
                    id='section-selector',
                    options=[{'label': str(s), 'value': s} for s in section_labels],
                    value=section_labels,
                    multi=True,
                    style={'marginBottom': '10px'}
                ),
            ], style={'marginBottom': '20px', 'padding': '10px', 'backgroundColor': '#e8f5e9', 'borderRadius': '5px'}),
            
            html.Div([
                html.Label("🎮 Operation mode:", style={'fontWeight': 'bold', 'marginBottom': '10px'}),
                dcc.RadioItems(
                    id='mode-selector',
                    options=[
                        {'label': ' Select temporary-point mode', 'value': 'select'},
                        {'label': ' Edit-midpoint mode', 'value': 'edit'}
                    ],
                    value='select',
                    style={'marginTop': '5px'}
                ),
                html.Div(id='mode-hint', style={'marginTop': '10px', 'padding': '8px', 'backgroundColor': '#e3f2fd', 'borderRadius': '5px', 'fontSize': '12px'})
            ], style={'marginBottom': '20px', 'padding': '10px', 'backgroundColor': '#f3e5f5', 'borderRadius': '5px'}),
            
            html.Div([
                html.Label(id='coord-box-label', children="Manually add a temporary point:", style={'fontWeight': 'bold'}), 
                html.Div([
                    dcc.Input(id='input-x', type='number', placeholder='X', style={'width': '60px', 'marginRight': '5px'}),
                    dcc.Input(id='input-y', type='number', placeholder='Y', style={'width': '60px', 'marginRight': '5px'}),
                    dcc.Input(id='input-z', type='number', placeholder='Z', style={'width': '60px', 'marginRight': '5px'}),
                    html.Button(id='action-btn', children='Add', n_clicks=0, style={'padding': '5px 10px'})
                ], style={'display': 'flex', 'marginBottom': '5px'}),
                html.Div(id='insert-position-selector', style={'marginTop': '5px'})
            ], style={'marginBottom': '20px', 'padding': '10px', 'backgroundColor': '#f0f0f0', 'borderRadius': '5px'}),
            
            html.Div([
                html.Label("🎯 Global coordinate-transform settings:", style={'fontWeight': 'bold', 'marginBottom': '10px'}),
                html.P("💡 Rotation and plane control of the frame have moved to the segment settings", style={'fontSize': '11px', 'color': '#666', 'marginBottom': '10px', 'fontStyle': 'italic'}),
                html.Div([
                    html.Label("Projection method:", style={'fontSize': '12px', 'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.RadioItems(
                        id='projection-method',
                        options=[
                            {'label': ' Based on the first midpoint (legacy method)', 'value': 'first_point'},
                            {'label': ' Based on the nearest plane (new method)', 'value': 'nearest_plane'}
                        ],
                        value='nearest_plane',
                        style={'fontSize': '11px'}
                    ),
                ], style={'marginBottom': '10px', 'padding': '10px', 'backgroundColor': '#fff3cd', 'borderRadius': '5px'}),
                html.Button('🔄 Apply coordinate transform', id='apply-transform-btn', n_clicks=0,
                           style={'width': '100%', 'padding': '8px', 'marginBottom': '10px',
                                 'backgroundColor': '#FF9800', 'color': 'white', 'border': 'none',
                                 'borderRadius': '5px', 'fontWeight': 'bold'}),
                html.Div([
                    html.Label("🎨 Colour coordinate:", style={'fontSize': '12px', 'fontWeight': 'bold', 'marginTop': '10px', 'marginBottom': '5px'}),
                    dcc.Dropdown(
                        id='color-coordinate',
                        options=[
                            {'label': 'New X coordinate (radial 1)', 'value': 'new_x'},
                            {'label': 'New Y coordinate (radial 2)', 'value': 'new_y'},
                            {'label': 'New Z coordinate (along the tube)', 'value': 'new_z'},
                            {'label': 'Original layer', 'value': 'layer'}
                        ],
                        value='new_y',
                        style={'fontSize': '11px'}
                    ),
                    
                    html.Div([
                        html.Hr(style={'margin': '10px 0'}),
                        html.Label("🎨 Colour-range control:", style={'fontSize': '11px', 'fontWeight': 'bold', 'marginBottom': '5px'}),
                        dcc.Checklist(
                            id='enable-color-range',
                            options=[{'label': ' Custom colour range', 'value': 'enabled'}],
                            value=[],
                            style={'fontSize': '10px', 'marginBottom': '8px'}
                        ),
                        html.Div(id='color-range-controls', children=[
                            html.Div([
                                html.Label("Min value:", style={'fontSize': '10px', 'marginRight': '5px', 'width': '50px'}),
                                dcc.Input(
                                    id='color-min-input',
                                    type='number',
                                    placeholder='auto',
                                    style={'width': '80px', 'marginRight': '10px', 'fontSize': '10px'}
                                ),
                                html.Label("Max value:", style={'fontSize': '10px', 'marginRight': '5px', 'width': '50px'}),
                                dcc.Input(
                                    id='color-max-input',
                                    type='number',
                                    placeholder='auto',
                                    style={'width': '80px', 'fontSize': '10px'}
                                ),
                            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '5px'}),
                            html.Div(id='color-range-info', style={'fontSize': '9px', 'color': '#666', 'fontStyle': 'italic'})
                        ], style={'display': 'none'}),  # Hidden by default
                    ], id='color-range-section'),
                ], style={'marginTop': '10px', 'padding': '10px', 'backgroundColor': '#f3e5f5', 'borderRadius': '5px'}),
                html.Button('💾 Export coordinate table', id='export-coords-btn', n_clicks=0,
                           style={'width': '100%', 'padding': '8px', 'marginTop': '10px',
                                 'backgroundColor': '#4CAF50', 'color': 'white', 'border': 'none',
                                 'borderRadius': '5px', 'fontWeight': 'bold'}),
            ], style={'marginBottom': '20px', 'padding': '10px', 'backgroundColor': '#e1f5fe', 'borderRadius': '5px'}),
            
            # Segment-parameter settings panel
            html.Div([
                html.Label("🎯 Segment coordinate settings:", style={'fontWeight': 'bold', 'fontSize': '14px', 'marginBottom': '10px', 'color': '#9C27B0'}),
                html.P("💡 Set different angles and radii for different midpoint segments; each segment can specify start and end", style={'fontSize': '11px', 'color': '#666', 'marginBottom': '5px', 'fontStyle': 'italic'}),
                html.P("✅ Coordinate mapping guarantees continuity: smooth transitions between adjacent segments", style={'fontSize': '11px', 'color': '#2e7d32', 'marginBottom': '10px', 'fontWeight': 'bold'}),
                
                html.Div(id='segments-controls-container', style={'marginBottom': '10px'}),
                
                html.Div([
                    html.Button('➕ Add segment', id='add-segment-btn', n_clicks=0, 
                               style={'width': '48%', 'padding': '6px', 'backgroundColor': '#9C27B0', 
                                     'color': 'white', 'border': 'none', 'borderRadius': '5px', 
                                     'fontSize': '12px', 'marginRight': '4%'}),
                    html.Button('🔄 Apply segments', id='apply-segments-btn', n_clicks=0,
                               style={'width': '48%', 'padding': '6px', 'backgroundColor': '#FF9800', 
                                     'color': 'white', 'border': 'none', 'borderRadius': '5px', 
                                     'fontSize': '12px'}),
                ], style={'display': 'flex', 'marginBottom': '10px'}),
                
                html.Div(id='segments-info', style={'fontSize': '11px', 'color': '#666', 'padding': '8px', 
                                                    'backgroundColor': '#fff9e6', 'borderRadius': '5px'}),
            ], style={'marginBottom': '20px', 'padding': '10px', 'backgroundColor': '#f3e5f5', 
                     'borderRadius': '5px', 'border': '2px solid #9C27B0'}),
            
            html.Div([
                html.Label("💾 Midpoint data management:", style={'fontWeight': 'bold', 'marginBottom': '10px'}),
                html.Div([
                    dcc.Upload(
                        id='upload-midpoints',
                        children=html.Button('📂 Import midpoints', style={'width': '48%', 'padding': '8px', 'backgroundColor': '#4CAF50', 'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
                        multiple=False
                    ),
                    html.Button('💾 Export midpoints', id='export-midpoints-btn', n_clicks=0, style={'width': '48%', 'padding': '8px', 'backgroundColor': '#2196F3', 'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer', 'marginLeft': '4%'}),
                ], style={'display': 'flex', 'justifyContent': 'space-between'})
            ], style={'marginBottom': '20px', 'padding': '10px', 'backgroundColor': '#e8f5e9', 'borderRadius': '5px'}),
            
            html.Div([
                html.Label("📋 Midpoint list (sortable):", style={'fontWeight': 'bold', 'marginBottom': '10px'}),
                html.Div(id='midpoints-list', style={'maxHeight': '120px', 'overflowY': 'auto', 'border': '1px solid #ddd', 'padding': '10px', 'backgroundColor': 'white', 'borderRadius': '5px'})
            ], style={'marginBottom': '20px', 'padding': '10px', 'backgroundColor': '#fff3e0', 'borderRadius': '5px'}),
            
            html.Div([html.Label("Cylinder radius:", style={'fontWeight': 'bold'}), dcc.Slider(id='radius-slider', min=0.5, max=200, step=0.5, value=2.5, marks={i: str(i) for i in [1, 10, 25, 50, 100, 150, 200]}, tooltip={"placement": "bottom", "always_visible": True})], style={'marginBottom': '20px'}),
            html.Div([html.Label("Cell point size:", style={'fontWeight': 'bold'}), dcc.Slider(id='point-size-slider', min=0.5, max=8, step=0.5, value=1.5, marks={i: str(i) for i in range(0, 9, 2)}, tooltip={"placement": "bottom", "always_visible": True})], style={'marginBottom': '20px'}),
            html.Div([html.Label("Radial segments:", style={'fontWeight': 'bold'}), dcc.Slider(id='segments-slider', min=8, max=32, step=4, value=20, marks={i: str(i) for i in [8, 16, 24, 32]}, tooltip={"placement": "bottom", "always_visible": True})], style={'marginBottom': '20px'}),
            html.Div([html.Label("Curve samples:", style={'fontWeight': 'bold'}), dcc.Slider(id='curve-samples-slider', min=50, max=1000, step=50, value=200, marks={i: str(i) for i in [50, 200, 400, 600, 800, 1000]}, tooltip={"placement": "bottom", "always_visible": True})], style={'marginBottom': '20px'}),
            html.Div([html.Label("Smoothing:", style={'fontWeight': 'bold'}), dcc.Slider(id='smoothing-slider', min=0, max=20, step=1, value=2, marks={i: str(i) for i in [0, 2, 5, 10, 15, 20]}, tooltip={"placement": "bottom", "always_visible": True})], style={'marginBottom': '20px'}),
            html.Div([html.Label("Display options:", style={'fontWeight': 'bold'}), dcc.Checklist(id='display-options', options=[{'label': ' Cells', 'value': 'cells'}, {'label': ' Midpoints', 'value': 'center_points'}, {'label': ' Temporary points', 'value': 'temp_points'}, {'label': ' Centerline', 'value': 'centerline'}, {'label': ' Cylinder', 'value': 'cylinder'}, {'label': ' Coordinate frame', 'value': 'coord_system'}], value=['cells', 'center_points', 'temp_points', 'centerline'])], style={'marginBottom': '20px'}),
            html.Div([
                html.Button('🎯 Generate midpoint', id='generate-midpoint-btn', n_clicks=0, style={'width': '100%', 'padding': '10px', 'marginBottom': '10px', 'backgroundColor': '#9C27B0', 'color': 'white', 'border': 'none', 'borderRadius': '5px', 'fontWeight': 'bold'}),
                html.Button('Clear temporary points', id='clear-temp-btn', n_clicks=0, style={'width': '100%', 'padding': '8px', 'marginBottom': '15px', 'backgroundColor': '#FF9800', 'color': 'white', 'border': 'none', 'borderRadius': '5px'}),
                html.Hr(),
                html.Button('Generate curve', id='generate-curve-btn', n_clicks=0, style={'width': '100%', 'padding': '10px', 'marginBottom': '10px', 'backgroundColor': '#4CAF50', 'color': 'white', 'border': 'none', 'borderRadius': '5px', 'fontWeight': 'bold'}),
                html.Button('Generate cylinder', id='generate-cylinder-btn', n_clicks=0, style={'width': '100%', 'padding': '10px', 'marginBottom': '10px', 'backgroundColor': '#2196F3', 'color': 'white', 'border': 'none', 'borderRadius': '5px', 'fontWeight': 'bold'}),
                html.Button('Reset all', id='reset-btn', n_clicks=0, style={'width': '100%', 'padding': '10px', 'marginBottom': '10px', 'backgroundColor': '#f44336', 'color': 'white', 'border': 'none', 'borderRadius': '5px'}),
            ]),
            html.Div(id='info-display', style={'marginTop': '20px', 'padding': '10px', 'backgroundColor': '#e8f5e9', 'borderRadius': '5px', 'fontSize': '13px'}),
        ], style={'width': '25%', 'padding': '20px', 'backgroundColor': '#f9f9f9', 'height': '100vh', 'overflowY': 'auto'}),
        html.Div([
            html.Div([
                dcc.Graph(id='3d-plot', style={'height': '50vh'}, config={'displayModeBar': True})
            ], style={'height': '50vh', 'borderBottom': '2px solid #ddd'}),
            html.Div([
                html.H4("🎨 Coordinate-mapping validation", style={'textAlign': 'center', 'margin': '10px', 'color': '#2196F3'}),
                dcc.Graph(id='validation-plot', style={'height': '45vh'}, config={'displayModeBar': True})
            ], style={'height': '50vh'})
        ], style={'width': '75%'}),
    ], style={'display': 'flex'}),
    dcc.Store(id='temp-points-store', data=[]),
    dcc.Store(id='center-points-store', data=[]),
    dcc.Store(id='selected-midpoint-index', data=None),
    dcc.Store(id='last-click-data', data=None),
    dcc.Store(id='centerline-store', data=None),
    dcc.Store(id='cylinder-store', data=None),
    dcc.Store(id='local-coord-system', data=None),
    dcc.Store(id='local-frames-store', data=None),
    dcc.Store(id='perpendicular-planes', data=None),
    dcc.Store(id='transformed-coords', data=None),
    dcc.Store(id='show-coord-system', data=False),
    dcc.Store(id='segments-store', data=[]),  # Store segment parameters
    dcc.Download(id='download-midpoints'),
    dcc.Download(id='download-coords'),
])

# Dynamically generate cluster colour controllers
# ======================
# Data loading and filtering callbacks
# ======================
# Legacy global-rotation callback removed; now fully segment-controlled
# Legacy plane-display callback removed; now segment-controlled

@app.callback(
    Output('transformed-coords', 'data'),
    Input('apply-transform-btn', 'n_clicks'),
    [State('local-frames-store', 'data'),
     State('projection-method', 'value')],
    prevent_initial_call=True
)
def apply_coordinate_transform(n_clicks, local_frames, projection_method):
    if not local_frames:
        print("⚠️ Please apply segment parameters first to generate the coordinate frame")
        return None
    
    cells = test_data['cells']
    if not cells:
        print("⚠️ No cell data")
        return None
    
    points = np.array([[c['x'], c['y'], c['z']] for c in cells])
    
    # Use the nearest-plane method (recommended in V7.0)
    new_coords, nearest_indices = transform_to_local_coords_nearest_plane(points, local_frames)
    
    transformed = []
    for i, cell in enumerate(cells):
        transformed.append({
            'id': cell['id'],
            'original_x': cell['x'],
            'original_y': cell['y'],
            'original_z': cell['z'],
            'new_x': float(new_coords[i, 0]),
            'new_y': float(new_coords[i, 1]),
            'new_z': float(new_coords[i, 2]),
            'nearest_curve_index': int(nearest_indices[i]),
            'layer': cell.get('layer', 'unknown'),
            'projection_method': projection_method
        })
    
    print(f"✅ Coordinate transform complete: {len(transformed)} cells")
    return transformed

@app.callback(
    Output('download-coords', 'data'),
    Input('export-coords-btn', 'n_clicks'),
    State('transformed-coords', 'data'),
    prevent_initial_call=True
)
def export_coordinate_mapping(n_clicks, transformed_coords):
    if not transformed_coords:
        return dash.no_update
    
    df = pd.DataFrame(transformed_coords)
    csv_string = df.to_csv(index=False, encoding='utf-8-sig')
    
    return dict(content=csv_string, filename="coordinate_mapping_v4.2.csv")

# [All other previous callback functions retained...]
@app.callback(Output('mode-hint', 'children'), Input('mode-selector', 'value'))
def update_mode_hint(mode):
    return "📍 Click the 3D plot to select 2 temporary points -> generate a midpoint" if mode == 'select' else "Click the edit icon to edit | use up/down to reorder"

@app.callback(Output('insert-position-selector', 'children'), [Input('center-points-store', 'data'), Input('mode-selector', 'value')])
def show_insert_selector(center_points, mode):
    # Simplified: no longer show the insert-position selector
    return None

@app.callback([Output('coord-box-label', 'children'), Output('action-btn', 'children'), Output('action-btn', 'style')], Input('selected-midpoint-index', 'data'))
def update_coord_box_ui(selected_idx):
    if selected_idx is not None: return (f"✏️ Edit midpoint{selected_idx+1}:", "Update", {'padding': '5px 10px', 'backgroundColor': '#FF9800', 'color': 'white', 'border': 'none', 'borderRadius': '5px', 'fontWeight': 'bold', 'cursor': 'pointer'})
    return ("Manually add a temporary point:", "Add", {'padding': '5px 10px'})

@app.callback([Output('input-x', 'value'), Output('input-y', 'value'), Output('input-z', 'value')], Input('selected-midpoint-index', 'data'), State('center-points-store', 'data'))
def fill_coords_on_select(selected_idx, center_points):
    if selected_idx is not None and center_points and 0 <= selected_idx < len(center_points):
        point = center_points[selected_idx]
        return point[0], point[1], point[2]
    return None, None, None

@app.callback(Output('center-points-store', 'data', allow_duplicate=True), Input('upload-midpoints', 'contents'), State('upload-midpoints', 'filename'), prevent_initial_call=True)
def import_midpoints(contents, filename):
    if contents is None: return dash.no_update
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string).decode('utf-8')
        data = json.loads(decoded)
        if 'midpoints' in data and isinstance(data['midpoints'], list):
            midpoints = data['midpoints']
            valid_points = [p for p in midpoints if isinstance(p, list) and len(p) == 3]
            if valid_points: return valid_points
    except: pass
    return dash.no_update

@app.callback(Output('download-midpoints', 'data'), Input('export-midpoints-btn', 'n_clicks'), State('center-points-store', 'data'), prevent_initial_call=True)
def export_midpoints(n_clicks, center_points):
    if not center_points: return dash.no_update
    return dict(content=json.dumps({'version': '1.0', 'description': 'tube midpoint data', 'midpoints': center_points, 'count': len(center_points)}, indent=2), filename="vessel_midpoints.json")

@app.callback([Output('temp-points-store', 'data'), Output('last-click-data', 'data')], [Input('action-btn', 'n_clicks'), Input('clear-temp-btn', 'n_clicks'), Input('generate-midpoint-btn', 'n_clicks'), Input('reset-btn', 'n_clicks'), Input('3d-plot', 'clickData')], [State('input-x', 'value'), State('input-y', 'value'), State('input-z', 'value'), State('temp-points-store', 'data'), State('last-click-data', 'data'), State('mode-selector', 'value'), State('selected-midpoint-index', 'data')])
def manage_temp_points(action_clicks, clear_clicks, midpoint_clicks, reset_clicks, click_data, x, y, z, temp_points, last_click, mode, selected_idx):
    ctx = callback_context
    if not ctx.triggered: return temp_points or [], last_click
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger_id in ['reset-btn', 'clear-temp-btn', 'generate-midpoint-btn']: return [], None
    if trigger_id == 'action-btn' and mode == 'select' and x is not None and y is not None and z is not None:
        temp_points = temp_points or []
        new_temp = (temp_points + [[float(x), float(y), float(z)]]) if len(temp_points) < 2 else [[float(x), float(y), float(z)]]
        return new_temp, last_click
    if trigger_id == '3d-plot' and click_data and mode == 'select':
        current_click_str = json.dumps(click_data, sort_keys=True)
        last_click_str = json.dumps(last_click, sort_keys=True) if last_click else None
        if current_click_str != last_click_str and 'points' in click_data and len(click_data['points']) > 0:
            point = click_data['points'][0]
            if 'x' in point and 'y' in point and 'z' in point:
                temp_points = temp_points or []
                new_point = [point['x'], point['y'], point['z']]
                new_temp = (temp_points + [new_point]) if len(temp_points) < 2 else [temp_points[1], new_point]
                return new_temp, click_data
    return temp_points or [], last_click

@app.callback([Output('center-points-store', 'data', allow_duplicate=True), Output('selected-midpoint-index', 'data', allow_duplicate=True)], [Input('generate-midpoint-btn', 'n_clicks'), Input('reset-btn', 'n_clicks'), Input({'type': 'delete-point', 'index': ALL}, 'n_clicks'), Input({'type': 'select-point', 'index': ALL}, 'n_clicks'), Input({'type': 'move-up', 'index': ALL}, 'n_clicks'), Input({'type': 'move-down', 'index': ALL}, 'n_clicks'), Input('action-btn', 'n_clicks')], [State('temp-points-store', 'data'), State('center-points-store', 'data'), State('selected-midpoint-index', 'data'), State('mode-selector', 'value'), State('input-x', 'value'), State('input-y', 'value'), State('input-z', 'value')], prevent_initial_call=True)
def manage_center_points(midpoint_clicks, reset_clicks, delete_clicks, select_clicks, move_up_clicks, move_down_clicks, action_clicks, temp_points, center_points, selected_idx, mode, x, y, z):
    ctx = callback_context
    if not ctx.triggered: return center_points or [], selected_idx
    trigger_id = ctx.triggered[0]['prop_id']
    if 'reset-btn' in trigger_id: return [], None
    if 'delete-point' in trigger_id:
        try:
            trigger_dict = json.loads(trigger_id.split('.')[0])
            idx_to_delete = trigger_dict['index']
            center_points = center_points or []
            if 0 <= idx_to_delete < len(center_points): return center_points[:idx_to_delete] + center_points[idx_to_delete+1:], None
        except: pass
    if 'move-up' in trigger_id:
        try:
            trigger_dict = json.loads(trigger_id.split('.')[0])
            idx = trigger_dict['index']
            center_points = center_points or []
            if idx > 0 and idx < len(center_points):
                new_points = center_points.copy()
                new_points[idx], new_points[idx-1] = new_points[idx-1], new_points[idx]
                return new_points, None
        except: pass
    if 'move-down' in trigger_id:
        try:
            trigger_dict = json.loads(trigger_id.split('.')[0])
            idx = trigger_dict['index']
            center_points = center_points or []
            if idx >= 0 and idx < len(center_points) - 1:
                new_points = center_points.copy()
                new_points[idx], new_points[idx+1] = new_points[idx+1], new_points[idx]
                return new_points, None
        except: pass
    if 'select-point' in trigger_id:
        try:
            trigger_dict = json.loads(trigger_id.split('.')[0])
            idx = trigger_dict['index']
            return center_points or [], None if selected_idx == idx else idx
        except: pass
    if 'generate-midpoint-btn' in trigger_id and temp_points and len(temp_points) == 2:
        midpoint = ((np.array(temp_points[0]) + np.array(temp_points[1])) / 2).tolist()
        center_points = center_points or []
        # Add to the end by default
        new_points = center_points + [midpoint]
        return new_points, None
    if 'action-btn' in trigger_id and mode == 'edit' and selected_idx is not None:
        if x is not None and y is not None and z is not None:
            center_points = center_points or []
            if 0 <= selected_idx < len(center_points):
                new_point = [float(x), float(y), float(z)]
                new_center_points = center_points[:selected_idx] + [new_point] + center_points[selected_idx+1:]
                return new_center_points, None
    return center_points or [], selected_idx

@app.callback(Output('midpoints-list', 'children'), [Input('center-points-store', 'data'), Input('selected-midpoint-index', 'data')])
def update_midpoints_list(center_points, selected_idx):
    if not center_points: return html.P("No midpoints yet", style={'color': 'gray', 'fontStyle': 'italic'})
    items = []
    for i, point in enumerate(center_points):
        is_selected = (i == selected_idx)
        up_btn = html.Button('⬆️', id={'type': 'move-up', 'index': i}, n_clicks=0, disabled=(i == 0), style={'padding': '2px 6px', 'fontSize': '12px', 'marginRight': '2px', 'backgroundColor': '#e3f2fd' if i > 0 else '#f5f5f5', 'border': '1px solid #2196f3' if i > 0 else '1px solid #ccc', 'borderRadius': '3px', 'cursor': 'pointer' if i > 0 else 'not-allowed'})
        down_btn = html.Button('⬇️', id={'type': 'move-down', 'index': i}, n_clicks=0, disabled=(i == len(center_points) - 1), style={'padding': '2px 6px', 'fontSize': '12px', 'marginRight': '5px', 'backgroundColor': '#e3f2fd' if i < len(center_points)-1 else '#f5f5f5', 'border': '1px solid #2196f3' if i < len(center_points)-1 else '1px solid #ccc', 'borderRadius': '3px', 'cursor': 'pointer' if i < len(center_points)-1 else 'not-allowed'})
        items.append(html.Div([html.Div([html.Span(f"Midpoint{i+1}: ", style={'fontWeight': 'bold', 'color': '#d32f2f', 'minWidth': '55px'}), html.Span(f"({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})", style={'fontSize': '11px'})], style={'flex': '1'}), html.Div([up_btn, down_btn, html.Button('✏️', id={'type': 'select-point', 'index': i}, n_clicks=0, style={'marginRight': '3px', 'padding': '2px 8px', 'fontSize': '12px', 'backgroundColor': '#ff9800' if is_selected else '#e3f2fd', 'color': 'white' if is_selected else 'black', 'border': f"2px solid {'#ff9800' if is_selected else '#2196f3'}", 'borderRadius': '3px', 'cursor': 'pointer', 'fontWeight': 'bold' if is_selected else 'normal'}), html.Button('❌', id={'type': 'delete-point', 'index': i}, n_clicks=0, style={'padding': '2px 8px', 'fontSize': '12px', 'backgroundColor': '#ffebee', 'border': '1px solid #ef5350', 'borderRadius': '3px', 'cursor': 'pointer'})])], style={'padding': '6px', 'marginBottom': '4px', 'backgroundColor': '#fff3e0' if is_selected else '#fff', 'border': f"2px solid {'#ff9800' if is_selected else '#ddd'}", 'borderRadius': '3px', 'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'}))
    return html.Div(items)

@app.callback(Output('centerline-store', 'data'), [Input('generate-curve-btn', 'n_clicks'), Input('center-points-store', 'data')], [State('curve-samples-slider', 'value'), State('smoothing-slider', 'value')])
def generate_centerline(n_clicks, center_points, curve_samples, smoothing):
    """Generate a smooth curve"""
    if center_points and len(center_points) >= 2:
        # Ensure parameters have sensible defaults
        n_samples = curve_samples if curve_samples else 200
        smooth_val = smoothing if smoothing is not None else 2
        
        print(f"📊 Generate curve: {len(center_points)} midpoints -> {n_samples} samples (smoothing={smooth_val})")
        
        curve = fit_smooth_curve(center_points, n_samples=n_samples, smoothing=smooth_val)
        return curve.tolist()
    return None

@app.callback(Output('cylinder-store', 'data'), Input('generate-cylinder-btn', 'n_clicks'), [State('centerline-store', 'data'), State('radius-slider', 'value'), State('segments-slider', 'value')], prevent_initial_call=True)
def generate_cylinder(n_clicks, centerline_data, radius, segments):
    if not centerline_data: return None
    vertices, faces = create_cylinder_with_rmf(np.array(centerline_data), radius=radius, radial_segments=segments or 20)
    return {'vertices': vertices.tolist(), 'faces': faces.tolist()}

@app.callback([Output('3d-plot', 'figure'), Output('info-display', 'children')], 
              [Input('temp-points-store', 'data'), Input('center-points-store', 'data'), Input('selected-midpoint-index', 'data'), 
               Input('centerline-store', 'data'), Input('cylinder-store', 'data'), Input('point-size-slider', 'value'), 
               Input('display-options', 'value'), Input('mode-selector', 'value'), Input('local-coord-system', 'data'), 
               Input('perpendicular-planes', 'data'), Input('transformed-coords', 'data'),
               Input('cluster-selector', 'value'), Input('section-selector', 'value')])
def update_plot(temp_points, center_points, selected_idx, centerline_data, cylinder_data, point_size, display_opts, mode, local_system, planes_data, transformed_coords, selected_clusters, selected_sections):
    fig = go.Figure()
    display_opts = display_opts or []
    selected_clusters = selected_clusters or []
    selected_sections = selected_sections or []
    
    # Load currently displayed cells from h5ad
    test_data['cells'] = []
    
    if 'cells' in display_opts and selected_clusters and selected_sections:
        for cluster in selected_clusters:
            mask = (mtx.obs['whole_leiden_str'] == cluster) & (mtx.obs['adjusted_cure_clustering'].isin(selected_sections))
            if mask.any():
                group = mtx.obs[mask]
                
                if not group.empty:
                    x_coords = group['x_kde_aligned'].values
                    y_coords = group['z_centroid'].values
                    z_coords = group['y_kde_aligned'].values
                    
                    # Add to test_data for coordinate transform
                    for cell_id, x, y, z in zip(group.index, x_coords, y_coords, z_coords):
                        # Safely get layer info (if present)
                        cell_layer = None
                        if 'layer' in mtx.obs.columns:
                            cell_layer = mtx.obs.loc[cell_id, 'layer'] if cell_id in mtx.obs.index else None
                        
                        test_data['cells'].append({
                            'id': cell_id,
                            'x': float(x),
                            'y': float(y),
                            'z': float(z),
                            'cluster': cluster,
                            'layer': cell_layer
                        })
                    
                    # Use the default colour
                    cluster_color = cluster_colors.get(cluster, '#808080')
                
                # Draw
                fig.add_trace(go.Scatter3d(
                    x=x_coords, y=y_coords, z=z_coords, mode='markers',
                    marker=dict(size=point_size, color=cluster_color, opacity=0.6),
                    name=f'{cluster}'
                ))
    
    cells = test_data['cells']
    if temp_points and 'temp_points' in display_opts:
        tp = np.array(temp_points)
        labels = ['Temp 1', 'Temp 2'][:len(temp_points)]
        fig.add_trace(go.Scatter3d(x=tp[:, 0], y=tp[:, 1], z=tp[:, 2], mode='markers+text+lines', marker=dict(size=12, color='gold', symbol='circle', line=dict(color='orange', width=2)), line=dict(color='gold', width=3), text=labels, textposition='top center', textfont=dict(size=14, color='orange', family='Arial Black'), name='Temporary points'))
    
    if center_points and 'center_points' in display_opts:
        cp = np.array(center_points)
        colors = ['orange' if i == selected_idx else 'red' for i in range(len(center_points))]
        fig.add_trace(go.Scatter3d(x=cp[:, 0], y=cp[:, 1], z=cp[:, 2], mode='markers+text+lines',marker=dict(size=12, color='red', symbol='circle', line=dict(color='red', width=2)), line=dict(color='red', width=3, dash='dot'), text=[f'{i+1}{"★" if i==selected_idx else ""}' for i in range(len(center_points))], textposition='top center', textfont=dict(size=14, color='red', family='Arial Black'), name='Midpoints'))
    
    if centerline_data and 'centerline' in display_opts:
        centerline = np.array(centerline_data)
        fig.add_trace(go.Scatter3d(x=centerline[:, 0], y=centerline[:, 1], z=centerline[:, 2], mode='lines', line=dict(color='blue', width=6), name='Fitted centerline'))
    
    if cylinder_data and 'cylinder' in display_opts:
        vertices, faces = np.array(cylinder_data['vertices']), np.array(cylinder_data['faces'])
        fig.add_trace(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], opacity=0.6, color='gold', name='Cylinder'))
    
    if local_system and 'coord_system' in display_opts:
        cross_x = np.array(local_system['cross_x'])
        fig.add_trace(go.Scatter3d(x=cross_x[:, 0], y=cross_x[:, 1], z=cross_x[:, 2], mode='lines', line=dict(color='red', width=6), name='X axis'))
        cross_y = np.array(local_system['cross_y'])
        fig.add_trace(go.Scatter3d(x=cross_y[:, 0], y=cross_y[:, 1], z=cross_y[:, 2], mode='lines', line=dict(color='green', width=6), name='Y axis'))
        circle = np.array(local_system['circle'])
        fig.add_trace(go.Scatter3d(x=circle[:, 0], y=circle[:, 1], z=circle[:, 2], mode='lines', line=dict(color='cyan', width=2, dash='dot'), name='Reference circle'))
        center = np.array(local_system['center'])
        fig.add_trace(go.Scatter3d(x=[center[0]], y=[center[1]], z=[center[2]], mode='markers+text', marker=dict(size=12, color='purple', symbol='x'), text=['Origin'], textposition='top center', name='CoordinateOrigin'))
    
    # Draw perpendicular planes (multi-segment support)
    if planes_data:
        # planes_data is now a list, each element representing one segment's planes
        if isinstance(planes_data, list):
            # New format: multi-segment planes
            for plane_set in planes_data:
                show_planes = plane_set.get('show_planes', [])
                seg_idx = plane_set.get('segment_index', 0)
                
                # Show the XZ plane (if selected for this segment)
                if 'xz' in show_planes:
                    xz_left = np.array(plane_set['xz_plane_left'])
                    xz_right = np.array(plane_set['xz_plane_right'])
                    n_points = len(xz_left)
                    
                    xz_vertices = np.vstack([xz_left, xz_right])
                    xz_x = xz_vertices[:, 0]
                    xz_y = xz_vertices[:, 1]
                    xz_z = xz_vertices[:, 2]
                    
                    xz_i, xz_j, xz_k = [], [], []
                    for idx in range(n_points - 1):
                        xz_i.extend([idx, idx + 1])
                        xz_j.extend([idx + 1, idx + n_points + 1])
                        xz_k.extend([idx + n_points, idx + n_points])
                    
                    fig.add_trace(go.Mesh3d(
                        x=xz_x, y=xz_y, z=xz_z,
                        i=xz_i, j=xz_j, k=xz_k,
                        opacity=0.3, color='red', 
                        name=f'Segment {seg_idx+1} - XZ plane',
                        showlegend=True
                    ))
                
                # Show the YZ plane (if selected for this segment)
                if 'yz' in show_planes:
                    yz_left = np.array(plane_set['yz_plane_left'])
                    yz_right = np.array(plane_set['yz_plane_right'])
                    n_points = len(yz_left)
                    
                    yz_vertices = np.vstack([yz_left, yz_right])
                    yz_x = yz_vertices[:, 0]
                    yz_y = yz_vertices[:, 1]
                    yz_z = yz_vertices[:, 2]
                    
                    yz_i, yz_j, yz_k = [], [], []
                    for idx in range(n_points - 1):
                        yz_i.extend([idx, idx + 1])
                        yz_j.extend([idx + 1, idx + n_points + 1])
                        yz_k.extend([idx + n_points, idx + n_points])
                    
                    fig.add_trace(go.Mesh3d(
                        x=yz_x, y=yz_y, z=yz_z,
                        i=yz_i, j=yz_j, k=yz_k,
                        opacity=0.3, color='green', 
                        name=f'Segment {seg_idx+1} - YZ plane',
                        showlegend=True
                    ))
        else:
            # Legacy format: single global plane (backward compatible)
            xz_left = np.array(planes_data['xz_plane_left'])
            xz_right = np.array(planes_data['xz_plane_right'])
            yz_left = np.array(planes_data['yz_plane_left'])
            yz_right = np.array(planes_data['yz_plane_right'])
            n_points = len(xz_left)
            
            # XZ plane
            xz_vertices = np.vstack([xz_left, xz_right])
            xz_x, xz_y, xz_z = xz_vertices[:, 0], xz_vertices[:, 1], xz_vertices[:, 2]
            xz_i, xz_j, xz_k = [], [], []
            for idx in range(n_points - 1):
                xz_i.extend([idx, idx + 1])
                xz_j.extend([idx + 1, idx + n_points + 1])
                xz_k.extend([idx + n_points, idx + n_points])
            
            fig.add_trace(go.Mesh3d(x=xz_x, y=xz_y, z=xz_z, i=xz_i, j=xz_j, k=xz_k,
                                   opacity=0.3, color='red', name='XZ plane', showlegend=True))
            
            # YZ plane
            yz_vertices = np.vstack([yz_left, yz_right])
            yz_x, yz_y, yz_z = yz_vertices[:, 0], yz_vertices[:, 1], yz_vertices[:, 2]
            yz_i, yz_j, yz_k = [], [], []
            for idx in range(n_points - 1):
                yz_i.extend([idx, idx + 1])
                yz_j.extend([idx + 1, idx + n_points + 1])
                yz_k.extend([idx + n_points, idx + n_points])
            
            fig.add_trace(go.Mesh3d(x=yz_x, y=yz_y, z=yz_z, i=yz_i, j=yz_j, k=yz_k,
                                   opacity=0.3, color='green', name='YZ plane', showlegend=True))
    
    fig.update_layout(
        scene=dict(
            aspectmode='data', 
            xaxis=dict(visible=False), 
            yaxis=dict(visible=False), 
            zaxis=dict(visible=False)
        ), 
        margin=dict(l=0, r=0, b=0, t=0), 
        showlegend=True, 
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'), 
        uirevision='constant', 
        dragmode='orbit',
        # High-resolution export settings
        width=1920,  # Export width (pixels)
        height=1080,  # Export height (pixels)
        font=dict(size=14)  # Increase font size for high resolution
    )
    
    mode_text = "📍 Select temporary-point mode" if mode == 'select' else "✏️ Edit-midpoint mode"
    info = [html.P([html.Strong(mode_text)], style={'color': '#1976d2'}), html.Hr(), html.P([html.Strong("Data:")]), html.P(f"• Cells: {len(test_data['cells'])}"), html.P(f"• Temporary points: {len(temp_points) if temp_points else 0}/2"), html.P(f"• Midpoints: {len(center_points) if center_points else 0}")]
    
    if transformed_coords:
        method = transformed_coords[0].get('projection_method', 'unknown')
        method_text = 'nearest plane' if method == 'nearest_plane' else 'first midpoint'
        info.append(html.P(f"✅ Transformed {len(transformed_coords)} points", style={'color': 'green', 'fontWeight': 'bold'}))
        info.append(html.P(f"📐 Projection method: {method_text}", style={'color': '#FF9800', 'fontSize': '11px'}))
    
    if mode == 'select':
        info.extend([html.Hr(), html.P([html.Strong("Select mode:")]), html.P("1️⃣ Select 2 temporary points"), html.P("2️⃣ Generate midpoint")])
    else:
        info.extend([html.Hr(), html.P([html.Strong("✏️ Edit mode:")]), html.P("✏️ Edit | ⬆️⬇️ Reorder")])
        if selected_idx is not None:
            info.append(html.P(f"🎯 Editing: midpoints{selected_idx+1}", style={'color': 'orange', 'fontWeight': 'bold'}))
    
    if temp_points and len(temp_points)==2 and mode=='select':
        info.append(html.P("✅ Ready to generate a midpoint!", style={'color': 'green', 'fontWeight': 'bold'}))
    if centerline_data:
        info.append(html.P("✓ Curve generated", style={'color': 'green'}))
    if cylinder_data:
        info.append(html.P("✓ Cylinder generated", style={'color': 'green'}))
    
    return fig, html.Div(info)

@app.callback(
    [Output('color-range-controls', 'style'),
     Output('color-range-info', 'children')],
    [Input('enable-color-range', 'value'),
     Input('color-coordinate', 'value'),
     Input('transformed-coords', 'data')]
)
def toggle_color_range_controls(enable_range, color_by, transformed_coords):
    """Control the display of the colour-range inputs and hint text"""
    if 'enabled' not in enable_range or color_by == 'layer':
        return {'display': 'none'}, ""
    
    # Compute the range of the current data
    if transformed_coords:
        if color_by == 'new_x':
            values = [c['new_x'] for c in transformed_coords]
            coord_name = "New X coordinate"
        elif color_by == 'new_y':
            values = [c['new_y'] for c in transformed_coords]
            coord_name = "New Y coordinate"
        else:  # new_z
            values = [c['new_z'] for c in transformed_coords]
            coord_name = "New Z coordinate"
        
        min_val = min(values)
        max_val = max(values)
        info = f"💡 Current {coord_name} range: [{min_val:.2f}, {max_val:.2f}]"
    else:
        info = "💡 Please apply the coordinate transform first to view the data range"
    
    return {'display': 'block'}, info

@app.callback(
    Output('validation-plot', 'figure'),
    [Input('transformed-coords', 'data'),
     Input('color-coordinate', 'value'),
     Input('point-size-slider', 'value'),
     Input('enable-color-range', 'value'),
     Input('color-min-input', 'value'),
     Input('color-max-input', 'value')]
)
def update_validation_plot(transformed_coords, color_by, point_size, enable_range, color_min, color_max):
    """Update the validation plot - colour the original point cloud with the new coordinates"""
    fig = go.Figure()
    
    if not transformed_coords:
        fig.add_annotation(
            text="Please apply the coordinate transform first",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False)
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            # High-resolution export settings
            width=1920,
            height=1080
        )
        return fig
    
    # Extract data
    x = [c['original_x'] for c in transformed_coords]
    y = [c['original_y'] for c in transformed_coords]
    z = [c['original_z'] for c in transformed_coords]
    
    # Colour according to the selected coordinate
    if color_by == 'layer':
        # Colour by original layer
        colors = [c.get('layer', 'unknown') for c in transformed_coords]
        color_discrete_map = {'upper': 'lightcoral', 'lower': 'cornflowerblue', 'unknown': 'gray'}
        
        # Draw by group
        for layer in ['upper', 'lower', 'unknown']:
            indices = [i for i, c in enumerate(colors) if c == layer]
            if indices:
                fig.add_trace(go.Scatter3d(
                    x=[x[i] for i in indices],
                    y=[y[i] for i in indices],
                    z=[z[i] for i in indices],
                    mode='markers',
                    marker=dict(
                        size=point_size,
                        color=color_discrete_map[layer],
                        opacity=0.6
                    ),
                    name=f"{'upper' if layer=='upper' else 'lower' if layer=='lower' else 'unknown'}"
                ))
        
        colorbar_title = "Layer"
    else:
        # Colour by the new coordinate
        if color_by == 'new_x':
            color_values = [c['new_x'] for c in transformed_coords]
            colorbar_title = "New X coordinate"
            colorscale = 'RdYlBu_r'
        elif color_by == 'new_y':
            color_values = [c['new_y'] for c in transformed_coords]
            colorbar_title = "New Y coordinate"
            colorscale = 'RdYlBu_r'
        else:  # new_z
            color_values = [c['new_z'] for c in transformed_coords]
            colorbar_title = "New Z coordinate"
            colorscale = 'Viridis'
        
        # Set the colour range
        marker_dict = {
            'size': point_size,
            'color': color_values,
            'colorscale': colorscale,
            'showscale': True,
            'colorbar': dict(
                title=colorbar_title,
                x=1.02
            ),
            'opacity': 0.6
        }
        
        # If a custom colour range is enabled
        if 'enabled' in enable_range:
            if color_min is not None:
                marker_dict['cmin'] = color_min
            if color_max is not None:
                marker_dict['cmax'] = color_max
        
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=marker_dict,
            name='Cells',
            hovertemplate='<b>Original coordinates</b><br>' +
                         'X: %{x:.2f}<br>' +
                         'Y: %{y:.2f}<br>' +
                         'Z: %{z:.2f}<br>' +
                         f'<b>{colorbar_title}</b>: %{{marker.color:.2f}}<extra></extra>'
        ))
    
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        margin=dict(l=0, r=0, b=20, t=20),
        showlegend=True,
        legend=dict(
            x=0.02, y=0.98,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        uirevision='constant',
        # High-resolution export settings
        width=1920,  # Export width (pixels)
        height=1080,  # Export height (pixels)
        font=dict(size=14)  # Increase font size for high resolution
    )
    
    return fig

# ======================
# Z-coordinate filtering callbacks
# ======================

# ======================
# Segment-parameter callbacks
# ======================

@app.callback(
    [Output('segments-controls-container', 'children'),
     Output('segments-info', 'children')],
    [Input('center-points-store', 'data'),
     Input('segments-store', 'data')]
)
def update_segments_controls(center_points, segments):
    """Dynamically generate the segment-parameter control interface"""
    if not center_points or len(center_points) < 2:
        return [html.P("⚠️ Please create at least 2 midpoints first", style={'color': '#999', 'fontStyle': 'italic', 'fontSize': '11px'})], \
               "Tip: once midpoints are created you can set segment parameters"
    
    n_points = len(center_points)
    
    # Initialize segments
    if not segments or len(segments) == 0:
        segments = [{'start_idx': 0, 'angle': 0.0, 'radius': 2.5}]
    
    controls = []
    for i, seg in enumerate(segments):
        controls.append(
            html.Div([
                # Header row
                html.Div([
                    html.Span(f"Segment {i+1}", style={'fontWeight': 'bold', 'color': '#9C27B0', 'fontSize': '12px'}),
                    html.Button('×', id={'type': 'delete-segment', 'index': i}, n_clicks=0,
                               style={'float': 'right', 'padding': '2px 6px', 'fontSize': '12px',
                                     'backgroundColor': '#f44336', 'color': 'white', 'border': 'none',
                                     'borderRadius': '3px', 'cursor': 'pointer'}) if len(segments) > 1 else None
                ], style={'marginBottom': '8px'}),
                
                # Start-midpoint selection
                html.Label(f"Start midpoint:", style={'fontSize': '11px', 'marginBottom': '3px'}),
                dcc.Dropdown(
                    id={'type': 'segment-start', 'index': i},
                    options=[{'label': f'Midpoint {j+1}', 'value': j} for j in range(n_points)],
                    value=seg.get('start_idx', 0),
                    clearable=False,
                    style={'marginBottom': '8px', 'fontSize': '11px'}
                ),
                
                # End-midpoint selection
                html.Label(f"End midpoint:", style={'fontSize': '11px', 'marginBottom': '3px'}),
                dcc.Dropdown(
                    id={'type': 'segment-end', 'index': i},
                    options=[{'label': f'Midpoint {j+1}', 'value': j} for j in range(n_points)],
                    value=seg.get('end_idx', n_points - 1),
                    clearable=False,
                    style={'marginBottom': '8px', 'fontSize': '11px'}
                ),
                
                # Rotation-angle control (mirrors the original local-frame design)
                html.Div([
                    html.Label("Rotation angle (deg):", style={'fontSize': '11px', 'marginBottom': '3px', 'fontWeight': 'bold'}),
                    html.Div([
                        dcc.Input(
                            id={'type': 'segment-angle-input', 'index': i},
                            type='number',
                            value=seg.get('angle', 0.0),
                            step=1,
                            style={'width': '50px', 'padding': '3px', 'fontSize': '10px', 'marginRight': '5px'}
                        ),
                        html.Span("°", style={'fontSize': '10px', 'marginRight': '5px'}),
                        html.Button('⟲ -45°', id={'type': 'segment-rotate-minus-45', 'index': i}, n_clicks=0,
                                   style={'padding': '3px 5px', 'fontSize': '9px', 'marginRight': '3px', 
                                         'border': '1px solid #ddd', 'borderRadius': '3px', 'cursor': 'pointer'}),
                        html.Button('⟲ -15°', id={'type': 'segment-rotate-minus-15', 'index': i}, n_clicks=0,
                                   style={'padding': '3px 5px', 'fontSize': '9px', 'marginRight': '3px',
                                         'border': '1px solid #ddd', 'borderRadius': '3px', 'cursor': 'pointer'}),
                        html.Button('⟳ +15°', id={'type': 'segment-rotate-plus-15', 'index': i}, n_clicks=0,
                                   style={'padding': '3px 5px', 'fontSize': '9px', 'marginRight': '3px',
                                         'border': '1px solid #ddd', 'borderRadius': '3px', 'cursor': 'pointer'}),
                        html.Button('⟳ +45°', id={'type': 'segment-rotate-plus-45', 'index': i}, n_clicks=0,
                                   style={'padding': '3px 5px', 'fontSize': '9px',
                                         'border': '1px solid #ddd', 'borderRadius': '3px', 'cursor': 'pointer'}),
                    ], style={'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap', 'marginBottom': '5px'}),
                    dcc.Slider(
                        id={'type': 'segment-angle-slider', 'index': i},
                        min=-180,
                        max=180,
                        step=5,
                        value=seg.get('angle', 0.0),
                        marks={j: str(j) for j in range(-180, 181, 45)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                ], style={'marginBottom': '8px', 'padding': '5px', 'backgroundColor': '#f5f5f5', 'borderRadius': '3px'}),
                
                # Perpendicular-plane display (independent per segment)
                html.Div([
                    html.Label("Perpendicular-plane display:", style={'fontSize': '10px', 'fontWeight': 'bold', 'marginBottom': '3px'}),
                    dcc.Checklist(
                        id={'type': 'segment-show-planes', 'index': i},
                        options=[
                            {'label': ' XZ plane (red)', 'value': 'xz'},
                            {'label': ' YZ plane (green)', 'value': 'yz'}
                        ],
                        value=seg.get('show_planes', []),
                        style={'fontSize': '9px'}
                    ),
                ], style={'marginBottom': '5px'}),
                
                # Plane-height control
                html.Div([
                    html.Label("Plane height:", style={'fontSize': '10px', 'fontWeight': 'bold', 'marginBottom': '3px'}),
                    dcc.Slider(
                        id={'type': 'segment-plane-height', 'index': i},
                        min=1,
                        max=500,
                        step=1,
                        value=seg.get('plane_height', None),
                        marks={j: str(j) for j in [50, 200, 500]},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.P("Tip: leave blank to use the default (radius x 2)", 
                           style={'fontSize': '8px', 'color': '#666', 'marginTop': '2px', 'fontStyle': 'italic'}),
                ], style={'marginBottom': '5px'}),
                
                # Radius control
                html.Div([
                    html.Label("Radius:", style={'fontSize': '11px', 'marginBottom': '3px'}),
                    dcc.Input(
                        id={'type': 'segment-radius', 'index': i},
                        type='number',
                        value=seg.get('radius', 2.5),
                        min=0.1,
                        step=0.1,
                        style={'width': '100%', 'padding': '4px', 'fontSize': '11px'}
                    ),
                ]),
                
                html.Hr(style={'margin': '10px 0'}) if i < len(segments) - 1 else None
            ], style={'padding': '8px', 'backgroundColor': '#fff', 'borderRadius': '5px', 'marginBottom': '8px',
                     'border': '1px solid #e0e0e0'})
        )
    
    # Info display
    info_text = f"✅ Currently {len(segments)} segments, {n_points} midpoints in total"
    if len(segments) > 0:
        sorted_segs = sorted(segments, key=lambda x: x.get('start_idx', 0))
        ranges = []
        for seg in sorted_segs:
            start = seg.get('start_idx', 0) + 1
            end = seg.get('end_idx', n_points - 1) + 1
            ranges.append(f"Midpoints {start}-{end}")
        info_text += f" | Segment ranges: {', '.join(ranges)}"
    
    return controls, info_text

@app.callback(
    Output('segments-store', 'data'),
    [Input('add-segment-btn', 'n_clicks'),
     Input({'type': 'delete-segment', 'index': ALL}, 'n_clicks'),
     Input({'type': 'segment-start', 'index': ALL}, 'value'),
     Input({'type': 'segment-end', 'index': ALL}, 'value'),
     Input({'type': 'segment-angle-input', 'index': ALL}, 'value'),
     Input({'type': 'segment-angle-slider', 'index': ALL}, 'value'),
     Input({'type': 'segment-rotate-minus-45', 'index': ALL}, 'n_clicks'),
     Input({'type': 'segment-rotate-minus-15', 'index': ALL}, 'n_clicks'),
     Input({'type': 'segment-rotate-plus-15', 'index': ALL}, 'n_clicks'),
     Input({'type': 'segment-rotate-plus-45', 'index': ALL}, 'n_clicks'),
     Input({'type': 'segment-radius', 'index': ALL}, 'value'),
     Input({'type': 'segment-show-planes', 'index': ALL}, 'value'),
     Input({'type': 'segment-plane-height', 'index': ALL}, 'value')],
    [State('segments-store', 'data'),
     State('center-points-store', 'data')],
    prevent_initial_call=True
)
def manage_segments(n_add, n_deletes, starts, ends, angle_inputs, angle_sliders, 
                   minus45_clicks, minus15_clicks, plus15_clicks, plus45_clicks,
                   radii, show_planes_list, plane_heights, segments, center_points):
    """Manage adding, deleting and updating segment parameters"""
    ctx = callback_context
    if not ctx.triggered:
        return segments or []
    
    trigger_id = ctx.triggered[0]['prop_id']
    
    # Initialize
    if not segments:
        segments = []
    
    # Add a new segment
    if 'add-segment-btn' in trigger_id:
        if center_points and len(center_points) > 0:
            # Find an unused start point
            used_starts = {seg.get('start_idx', 0) for seg in segments}
            new_start = 0
            for i in range(len(center_points)):
                if i not in used_starts:
                    new_start = i
                    break
            
            segments.append({
                'start_idx': new_start,
                'end_idx': len(center_points) - 1,  # default to the last midpoint
                'angle': 0.0,
                'radius': 2.5,
                'show_planes': [],
                'plane_height': None
            })
        return segments
    
    # Delete a segment
    if 'delete-segment' in trigger_id:
        if len(segments) > 1:
            button_id = json.loads(trigger_id.split('.')[0])
            idx = button_id['index']
            if 0 <= idx < len(segments):
                segments.pop(idx)
        return segments
    
    # Handle rotation buttons
    if 'segment-rotate' in trigger_id:
        button_id = json.loads(trigger_id.split('.')[0])
        idx = button_id['index']
        
        if 0 <= idx < len(segments):
            current_angle = segments[idx].get('angle', 0.0)
            
            if 'minus-45' in trigger_id:
                new_angle = current_angle - 45
            elif 'minus-15' in trigger_id:
                new_angle = current_angle - 15
            elif 'plus-15' in trigger_id:
                new_angle = current_angle + 15
            elif 'plus-45' in trigger_id:
                new_angle = current_angle + 45
            else:
                new_angle = current_angle
            
            # Normalize the angle to -180..180
            new_angle = ((new_angle + 180) % 360) - 180
            segments[idx]['angle'] = new_angle
        
        return segments
    
    # Update all segment parameters
    if starts and ends and angle_inputs and radii:
        new_segments = []
        for i in range(min(len(starts), len(ends), len(angle_inputs), len(radii))):
            # Decide whether to use the input or slider value (use the most recently updated one)
            angle = angle_inputs[i] if angle_inputs[i] is not None else 0.0
            if angle_sliders and i < len(angle_sliders) and angle_sliders[i] is not None:
                # Check whether the slider was updated
                if 'segment-angle-slider' in trigger_id:
                    angle = angle_sliders[i]
            
            start_idx = starts[i] if starts[i] is not None else 0
            end_idx = ends[i] if ends[i] is not None else len(center_points) - 1
            
            # Validate: ensure start_idx <= end_idx
            if start_idx > end_idx:
                end_idx = start_idx  # If start > end, adjust the end automatically
            
            new_segments.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'angle': angle,
                'radius': radii[i] if radii[i] is not None else 2.5,
                'show_planes': show_planes_list[i] if show_planes_list and i < len(show_planes_list) else [],
                'plane_height': plane_heights[i] if plane_heights and i < len(plane_heights) else None
            })
        
        # Sort by start_idx
        new_segments.sort(key=lambda x: x['start_idx'])
        return new_segments
    
    return segments

@app.callback(
    [Output('centerline-store', 'data', allow_duplicate=True),
     Output('local-frames-store', 'data', allow_duplicate=True),
     Output('perpendicular-planes', 'data', allow_duplicate=True)],
    Input('apply-segments-btn', 'n_clicks'),
    [State('center-points-store', 'data'),
     State('segments-store', 'data'),
     State('smoothing-slider', 'value'),
     State('curve-samples-slider', 'value')],
    prevent_initial_call=True
)
def apply_segmented_parameters(n_clicks, center_points, segments, smoothing, n_samples):
    """Apply segment parameters and regenerate the curve and coordinate frame"""
    if not center_points or len(center_points) < 2:
        return dash.no_update, dash.no_update, dash.no_update
    
    if not segments or len(segments) == 0:
        # If there are no segments, use default parameters
        segments = [{'start_idx': 0, 'end_idx': len(center_points)-1, 'angle': 0.0, 'radius': 2.5, 'show_planes': [], 'plane_height': None}]
    
    # Generate a smooth curve
    centerline = fit_smooth_curve(center_points, n_samples=n_samples, smoothing=smoothing)
    
    # Compute local frames using segment parameters
    local_frames = compute_local_frames_along_curve_segmented(centerline, center_points, segments)
    
    # Generate planes for each segment (based on the segment start)
    all_planes = []
    n_center_points = len(center_points)
    n_curve_points = len(centerline)
    
    for seg_idx, seg in enumerate(segments):
        start_idx = seg.get('start_idx', 0)
        end_idx = seg.get('end_idx', n_center_points - 1)
        show_planes = seg.get('show_planes', [])
        plane_height = seg.get('plane_height', None)
        
        # Generate planes only for segments that should display them
        if show_planes and len(show_planes) > 0:
            # Compute the segment start position on the curve
            curve_start_idx = int((start_idx / (n_center_points - 1)) * (n_curve_points - 1))
            curve_end_idx = int((end_idx / (n_center_points - 1)) * (n_curve_points - 1))
            
            # Create a local frame for this segment
            seg_local_system = create_local_coordinate_system(
                centerline[curve_start_idx],
                centerline[curve_start_idx:curve_end_idx+1],
                seg['radius'],
                seg['angle']
            )
            
            # Generate this segment's planes (only along its portion of the curve)
            seg_centerline = centerline[curve_start_idx:curve_end_idx+1]
            seg_planes = create_perpendicular_planes(
                seg_local_system, 
                seg_centerline, 
                seg['radius'],
                plane_height
            )
            
            all_planes.append({
                'segment_index': seg_idx,
                'show_planes': show_planes,
                'xz_plane_left': seg_planes['xz_plane_left'].tolist(),
                'xz_plane_right': seg_planes['xz_plane_right'].tolist(),
                'yz_plane_left': seg_planes['yz_plane_left'].tolist(),
                'yz_plane_right': seg_planes['yz_plane_right'].tolist()
            })
    
    print(f"✅ Apply segmentsParameters: {len(segments)}  segments")
    for i, seg in enumerate(sorted(segments, key=lambda x: x['start_idx'])):
        start = seg.get('start_idx', 0) + 1
        end = seg.get('end_idx', len(center_points) - 1) + 1
        show = seg.get('show_planes', [])
        print(f"  Segment {i+1}: midpoints {start}-{end}, angle={seg['angle']} deg, radius={seg['radius']}, planes={show}")
    
    return centerline.tolist(), {
        'centers': local_frames['centers'].tolist(),
        'tangents': local_frames['tangents'].tolist(),
        'normals': local_frames['normals'].tolist(),
        'binormals': local_frames['binormals'].tolist()
    }, all_planes

if __name__ == '__main__':
    print("\n🎯 Tube visualization tool V6.0 - enhanced edition")
    print("=" * 60)
    print("✅ Existing features (fully retained):")
    print("   - Projection based on the nearest plane")
    print("   - Resolves projection issues in curved regions")
    print("   - Comparison of two projection methods")
    print("   - Coordinate-mapping validation view")
    print("\n🆕 New feature 1: independent coordinate control per segment")
    print("   - Set the rotation angle independently per segment (-180 to +180 deg)")
    print("   - Shortcut buttons: -45, -15, +15, +45 deg")
    print("   - Display perpendicular planes (XZ/YZ) independently per segment")
    print("   - Set plane height and radius independently per segment")
    print("   - Parameters transition smoothly between segments")
    print("\n🆕 New feature 2: click-to-select filtering by Z value")
    print("   - List all unique z_centroid values")
    print("   - Click checkboxes to select which Z values to show")
    print("   - Select-all / clear shortcut buttons")
    print("   - Show the selected count in real time")
    print("\n" + "=" * 60)
    print("🌐 Address: http://127.0.0.1:8050")
    print("📖 Usage: see the accompanying documentation\n")
    app.run(debug=True, port=8050)