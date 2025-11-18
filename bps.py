"""
Basis Points Set Tools
for ADLR Trajectory Planning Project
Moritz Schüler and Alexander João Peterson Santos
2025-11-10
"""

import numpy as np
from typing import List, Tuple, Dict, Union
from sklearn.neighbors import BallTree

# constants
PRECISION = np.float32

"""
BPS FUNCTIONS
"""

def create_point_cloud(occupancy_grid: np.ndarray) -> np.ndarray:
    """
    Create point cloud based on the occupancy grid of a scene.

    Parameters
    ----------
    occupancy_grid: np.ndarray
        2d or 3d grid of non-zero values for points

    Returns
    -------
    np.ndarray
        2d or 3d points of point cloud
    """
    if not isinstance(occupancy_grid, np.ndarray) or occupancy_grid.ndim not in [2, 3]:
        raise ValueError("occupancy_grid is not a 2d or 3d np.ndarray")
    point_cloud_arr: np.ndarray = np.argwhere(occupancy_grid != 0).astype(PRECISION)
    return point_cloud_arr

def generate_bps(num_points: int, num_dims: int, radius: float = 1.0, random_seed: int = 13) -> np.ndarray:
    """
    Generate a Basis Points Set by sampling uniformly from a unit circle or sphere.

    References
    ----------
    Code adapted from Sergey Prokudin, Christoph Lassner, Javier Romero
    https://github.com/amzn/basis-point-sets/blob/master/bps/bps.py

    Parameters
    ----------
    num_points: int
        number of basis points to sample (k)
    num_dims: int
        dimensions (either 2d or 3d points)
    radius: float
        radius in which to sample points, max 1
    random_seed: int
        random seed for sampling

    Returns
    -------
    np.ndarray
        2d or 3d point coords
    """
    if num_dims not in [2, 3]:
        raise ValueError("num_dims must be 2 or 3")
    if radius <= 0 or radius > 1:
        raise ValueError("radius must be in interval (0, 1]")
    if num_points <= 0:
        raise ValueError("num_points must be at least 1")

    np.random.seed(random_seed)
    # sample point from d-sphere
    x = np.random.normal(size=[num_points, num_dims])
    x_norms = np.sqrt(np.sum(np.square(x), axis=1)).reshape([-1, 1])
    x_unit = x / x_norms

    # now sample radiuses uniformly
    r = np.random.uniform(size=[num_points, 1])
    u = np.power(r, 1.0 / num_dims)
    x = radius * x_unit * u
    np.random.seed(None)

    return x

def encode_scene(scene_point_cloud: np.ndarray, basis_point_cloud: np.ndarray, encoding: str) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """

    Parameters
    ----------
    scene_point_cloud
    basis_point_cloud
    encoding: str (either "scalar" or "diff" or "both")

    Returns
    -------

    """
    normalized_scene_point_cloud: np.ndarray = _normalize_scene(scene_point_cloud)
    nn_distances, nn_indexes = _nearest_neighbor_query(normalized_scene_point_cloud, basis_point_cloud)

    if encoding == "scalar":
        return nn_distances
    elif encoding == "diff":
        return _calculate_diff_vectors(normalized_scene_point_cloud, basis_point_cloud, nn_indexes)
    elif encoding == "both":
        return (nn_distances, _calculate_diff_vectors(normalized_scene_point_cloud, basis_point_cloud, nn_indexes))
    else:
        raise ValueError("encoding should be one of 'scalar', 'diff', or 'both'")




"""
HELPER FUNCTIONS
"""

def _normalize_scene(scene_point_cloud: np.ndarray) -> np.ndarray:
    """
    Normalize the scene such that it fits in a unit ball, i.e. no point is more than 1 unit from center.

    Parameters
    ----------
    scene_point_cloud: np.ndarray
        2d or 3d points in scene

    Returns
    -------
    np.ndarray
        2d or 3d points in normalized scene
    """
    # TODO test case to make sure every point fits in unit cloud
    centroid: np.ndarray = np.mean(scene_point_cloud, axis=0).astype(PRECISION)
    centered_cloud: np.ndarray = scene_point_cloud - centroid
    distances_from_centroid: np.ndarray = np.linalg.norm(centered_cloud, axis=1)
    max_distance: float = np.max(distances_from_centroid)
    normalized_scene_point_cloud: np.ndarray = centered_cloud / max_distance
    return normalized_scene_point_cloud

def _nearest_neighbor_query(scene_point_cloud: np.ndarray, basis_point_cloud: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get distances and indices of nearest neighbors to basis points in scene.

    Parameters
    ----------
    scene_point_cloud: np.ndarray
        2d or 3d points in the scene
    basis_point_cloud: np.ndarray
        2d or 3d points in a BPS

    Returns
    -------
    distances: np.ndarray
        distances to nearest neighbors
    indexes: np.ndarray
        indexes of points in scene_point_cloud which are closest to points in basis_point_cloud
    """
    # create ball tree of scene w/ Euclidean distances and query it for nearest neighbors
    scene_ball_tree: BallTree = BallTree(scene_point_cloud, leaf_size=40, metric="minkowski")
    distances, indexes = scene_ball_tree.query(basis_point_cloud, k=1)
    return (distances, indexes)

def _calculate_diff_vectors(scene_point_cloud: np.ndarray, basis_point_cloud: np.ndarray, nn_indexes: np.ndarray) -> np.ndarray:
    """
    Given indexes for nearest-neighbor matching (see _nearest_neighbor_query func), compute difference vectors.

    Parameters
    ----------
    scene_point_cloud : ndarray
        2d or 3d points in scene
    basis_point_cloud : ndarray
        3d or 3d points from BPS
    nn_indexes : ndarray
        indexes from _nearest_neighbor_query

    Returns
    -------
    np.ndarray
        Difference vectors from basis points to nearest points in scene
    """
    closest_scene_points: np.ndarray = scene_point_cloud[nn_indexes.squeeze()]
    diff_vecs: np.ndarray = closest_scene_points - basis_point_cloud
    return diff_vecs











