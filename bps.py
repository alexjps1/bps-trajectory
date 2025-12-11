"""
Basis Points Set Tools
for ADLR Trajectory Planning Project
Moritz Schüler and Alexander João Peterson Santos
2025-11-10
"""

# standard library imports
from typing import cast

# third party imports
import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import BallTree


"""
BPS FUNCTIONS
"""

def create_point_cloud(occupancy_grid: NDArray[np.int64]) -> NDArray[np.float64]:
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
    if not occupancy_grid.ndim not in [2, 3]:
        raise ValueError("occupancy_grid is not a 2d or 3d np.ndarray")
    mask: NDArray[np.bool_] = cast(NDArray[np.bool_], occupancy_grid != 0)
    point_cloud_arr: NDArray[np.float64] = np.argwhere(mask).astype(np.float64)
    return point_cloud_arr


def generate_bps_sampling(
    num_points: int,
    num_dims: int,
    shape: str,
    radius: float = 1.0,
    random_seed: int = 13,
) -> NDArray[np.float64]:
    """
    Generate a Basis Points Set by sampling uniformly from a square, cube, circle, or sphere.

    Parameters
    ----------
    num_points: int
        number of basis points to sample (k)
    num_dims: int (2 or 3)
        dimensionality of space in which to sample
    shape: str ("nsphsere" or "ncube")
        shape in which to sample
    radius: float
        radius in which to sample points, max 1
        for an ncube, this is the apothem
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

    if shape == "ncube":
        return _random_sampling_ncube(num_points, num_dims, radius, random_seed)
    elif shape == "nsphere":
        return _random_sampling_nsphere(num_points, num_dims, radius, random_seed)
    else:
        raise ValueError("shape must be 'ncube' or 'nsphere'")


def generate_bps_ngrid(
    grid_size: int,
    num_dims: int,
    minv: float = -1.0,
    maxv: float = 1.0,
) -> NDArray[np.float64]:
    """
    Generate n-dimensional grid Basis Points Set.

    References
    ----------
    Code adapted from Sergey Prokudin, Christoph Lassner, Javier Romero
    https://github.com/amzn/basis-point-sets/blob/master/bps/bps.py

    Parameters
    ----------
    grid_size: int
        number of elements in each grid axe
    num_dims: int
        number of dimensions for grid (2 or 3)
    minv: float
        minimum element of the grid
    maxv: float
        maximum element of the grid

    Returns
    -------
    basis: np.ndarray
        generated grid basis point set, a 2d ndarray containing point coordinates
        shape (grid_size**2, 2) if num_dims is 2
        shape (grid_size**2, 3) if num_dims is 3

    """
    if num_dims not in [2, 3]:
        raise ValueError("num_dims must be 2 or 3")

    linspaces = [np.linspace(minv, maxv, num=grid_size) for _ in range(0, num_dims)]
    coords = np.meshgrid(*linspaces)
    basis = np.concatenate(
        [coords[i].reshape([-1, 1]) for i in range(0, num_dims)], axis=1
    )

    return basis


def encode_scene(
    scene_point_cloud: NDArray[np.float64],
    basis_point_cloud: NDArray[np.float64],
    encoding_type: str,
    norm_bound_shape: str,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Returns BPS-encoded ndarray of a scene.

    Parameters
    ----------
    scene_point_cloud: np.ndarray
        Array of 2d or 3d points of the scene
    basis_point_cloud: np.ndarray
        Array of 3d or 3d points of a Basis Points Set
    encoding_type: str "scalar" or "diff" or "both"
        "scalar" gives distances to nearest neighbors
        "diff" dives difference vectors (2d or 3d)
        "both" gives a tuple like (distances, diffvectors)
    norm_bound_shape: str "ncube" or "nsphere" or "none"
        which shape to use for scene normalization
        use ncube with grid-based BPS or nsphere with random sampling-based BPS
        use none for no normalization at all (don't use this for ML-based scene reconstruction)


    Returns
    -------
    np.ndarray or Tuple[np.ndarray, np.ndarray]
        The BPS-encoded scene
    """
    normalized_scene_point_cloud: NDArray[np.float64] = _normalize_scene(
        scene_point_cloud, norm_bound_shape
    )
    nn_distances: NDArray[np.float64]
    nn_indexes: NDArray[np.int64]
    nn_distances, nn_indexes = _nearest_neighbor_query(
        normalized_scene_point_cloud, basis_point_cloud
    )

    if encoding_type == "scalar":
        return nn_distances
    elif encoding_type == "diff":
        return _calculate_diff_vectors(
            normalized_scene_point_cloud, basis_point_cloud, nn_indexes
        )
    elif encoding_type == "both":
        return (
            nn_distances,
            _calculate_diff_vectors(
                normalized_scene_point_cloud, basis_point_cloud, nn_indexes
            ),
        )
    else:
        raise ValueError("encoding should be one of 'scalar', 'diff', or 'both'")


"""
HELPER FUNCTIONS
"""


def _normalize_scene(
    scene_point_cloud: NDArray[np.float64], bound_shape: str
) -> NDArray[np.float64]:
    """
    Normalize the scene by centering points and scaling to fit within a unit n-sphere or unit n-cube.

    Parameters
    ----------
    scene_point_cloud: np.ndarray
        2d or 3d points in scene
    bound_shape: str "ncube" or "nsphere" or "none"
        which shape to use for scene normalization
        use ncube with grid-based BPS or nsphere with random sampling-based BPS
        do not use "none" for ML-based scene reconstruction, only for procedural

    Returns
    -------
    np.ndarray
        2d or 3d points in normalized scene
    """
    if bound_shape == "none":
        return scene_point_cloud.astype(np.float64)

    centroid: NDArray[np.float64] = cast(
        NDArray[np.float64], np.mean(scene_point_cloud, axis=0)
    )
    centered_cloud: NDArray[np.float64] = scene_point_cloud - centroid

    max_distance: np.float64
    if bound_shape == "nsphere":
        distances_from_centroid: NDArray[np.float64] = cast(
            NDArray[np.float64], np.linalg.norm(centered_cloud, axis=1)
        )
        max_distance = np.max(distances_from_centroid)
    elif bound_shape == "ncube":
        max_distance = cast(np.float64, np.max(np.abs(centered_cloud)))
    else:
        raise ValueError("bound_shape should be 'nsphere' or 'ncube'")

    normalized_scene_point_cloud: NDArray[np.float64] = centered_cloud / max_distance
    return normalized_scene_point_cloud


def _nearest_neighbor_query(
    scene_point_cloud: NDArray[np.float64], basis_point_cloud: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
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
    scene_ball_tree: BallTree = BallTree(
        scene_point_cloud, leaf_size=40, metric="minkowski"
    )
    distances: NDArray[np.float64]
    indexes: NDArray[np.int64]
    distances, indexes = cast(
        tuple[NDArray[np.float64], NDArray[np.int64]],
        scene_ball_tree.query(basis_point_cloud, k=1),
    )
    return (distances, indexes)


def _calculate_diff_vectors(
    scene_point_cloud: NDArray[np.float64],
    basis_point_cloud: NDArray[np.float64],
    nn_indexes: NDArray[np.int64],
) -> NDArray[np.float64]:
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
    closest_scene_points: NDArray[np.float64] = scene_point_cloud[nn_indexes.squeeze()]
    diff_vecs: NDArray[np.float64] = closest_scene_points - basis_point_cloud
    return diff_vecs


def _random_sampling_nsphere(
    num_points: int, num_dims: int, radius: float = 1.0, random_seed: int = 13
) -> NDArray[np.float64]:
    """
    Get points by sampling uniformly from an n-sphere.

    References
    ----------
    Code adapted from Sergey Prokudin, Christoph Lassner, Javier Romero
    https://github.com/amzn/basis-point-sets/blob/master/bps/bps.py

    Parameters
    ----------
    num_points: int
        number of points to sample
    num_dims: int
        dimensionality of space in which to sample
    radius: float
        radius in which to sample points
    random_seed: int
        random seed for sampling

    Returns
    -------
    np.ndarray
        coords of sampled points
    """
    if num_points <= 0:
        raise ValueError("num_points must be at least 1")

    np.random.seed(random_seed)
    # sample point from d-sphere
    x = np.random.normal(size=[num_points, num_dims])
    x_norms = cast(
        NDArray[np.float64], np.sqrt(np.sum(np.square(x), axis=1)).reshape([-1, 1])
    )
    x_unit = x / x_norms

    # now sample radiuses uniformly
    r = np.random.uniform(size=[num_points, 1])
    u = np.power(r, 1.0 / num_dims)
    x = radius * x_unit * u
    np.random.seed(None)

    return x


def _random_sampling_ncube(
    num_points: int, num_dims: int, apothem: float = 1.0, random_seed: int = 13
) -> NDArray[np.float64]:
    """
    Get points by sampling uniformly from an n-cube.

    Parameters
    ----------
    num_points: int
        number of points to sample
    num_dims: Literal[2, 3]
        dimensionality of space in which to sample
    apothem: float
        distance from the center to the mid-point of a side of the square (like radius)
        points are sampled inside an n-cube defined by this apothem
    random_seed: int
        Random seed for sampling

    Returns
    -------
    np.ndarray
        coords of sampled points
    """
    if num_points <= 0:
        raise ValueError("num_points must be at least 1")

    np.random.seed(random_seed)
    x = np.random.uniform(low=-apothem, high=apothem, size=(num_points, num_dims))
    np.random.seed(None)

    return x
