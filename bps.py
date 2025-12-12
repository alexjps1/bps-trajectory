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


def create_scene_point_cloud(
    occupancy_grid: NDArray[np.int64], create_empty_cloud: bool = False
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Creates two point clouds based on the occupancy grid of a scene.
    The occupied point cloud contains points where the occupancy grid is non-zero.
    The empty point cloud contains points where the occupancy grid is zero.

    Parameters
    ----------
    occupancy_grid: np.ndarray
        2d or 3d grid of non-zero values for points
    create_empty_cloud: bool
        Whether to create an empty point cloud

    Returns
    -------
    occupied_point_cloud: np.ndarray
        2d or 3d points of occupied point cloud
    empty_point_cloud: np.ndarray
        2d or 3d points of empty point cloud
    """
    if occupancy_grid.ndim not in [2, 3]:
        raise ValueError("occupancy_grid is not a 2d or 3d np.ndarray")
    mask: NDArray[np.bool_] = cast(NDArray[np.bool_], occupancy_grid != 0)
    occupied_point_cloud_arr: NDArray[np.float64] = np.argwhere(mask).astype(np.float64)
    if not create_empty_cloud:
        return occupied_point_cloud_arr
    empty_point_cloud_arr: NDArray[np.float64] = np.argwhere(~mask).astype(np.float64)
    return occupied_point_cloud_arr, empty_point_cloud_arr


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
    basis = np.concatenate([coords[i].reshape([-1, 1]) for i in range(0, num_dims)], axis=1)

    return basis


def encode_scene(
    scene_point_cloud: NDArray[np.float64],
    basis_point_cloud: NDArray[np.float64],
    encoding_type: str,
    norm_bound_shape: str,
    scene_empty_point_cloud: None | NDArray[np.float64] = None,
    grid_shape_for_grid_basis: None | tuple[int, int] = None,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Returns BPS-encoded ndarray of a scene.
    If scene_empty_point_cloud is provided, a signed BPS encoding is used (only supported for scalar encoding).
    This means that the encoding will contain negative values when basis points are inside an object.

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
    scene_empty_point_cloud: np.ndarray (optional)
        Array of 2d or 3d points representing empty space in the scene
        If this is provided, the encoding will contain negative values when basis points are inside an object
    grid_shape_for_grid_basis: tuple[int, int] (optional)
        Only relevant if the basis_point_cloud is grid-based and encoding_type == "scalar"
        Pass the shape of the bps grid here and the resulting bps encoding will be reshaped to be grid-shaped
        Ideal for working with convolutional models, which can use this for upsampling

    Returns
    -------
    np.ndarray or Tuple[np.ndarray, np.ndarray]
        The BPS-encoded scene
    """
    if scene_empty_point_cloud is not None and encoding_type != "scalar":
        raise ValueError("scene_empty_point_cloud can only be used with encoding type 'scalar'")

    # normalize the scene cloud and the empty cloud (if provided)
    normalized_scene_point_cloud: NDArray[np.float64]
    normalized_scene_empty_point_cloud: None | NDArray[np.float64] = None
    if scene_empty_point_cloud is not None:
        normalized_scene_point_cloud, normalized_scene_empty_point_cloud = cast(
            tuple[NDArray[np.float64], NDArray[np.float64]],
            _normalize_scene(scene_point_cloud, norm_bound_shape, scene_empty_point_cloud),
        )
    else:
        normalized_scene_point_cloud = cast(NDArray[np.float64], _normalize_scene(scene_point_cloud, norm_bound_shape))

    # do a nearest neighbor query for the scene cloud
    nn_distances: NDArray[np.float64]
    nn_indexes: NDArray[np.int64]
    nn_distances, nn_indexes = _nearest_neighbor_query(normalized_scene_point_cloud, basis_point_cloud)

    # do a nearest neighbor query for the empty cloud (if provided)
    if normalized_scene_empty_point_cloud is not None:
        empty_nn_distances: NDArray[np.float64]
        empty_nn_indexes: NDArray[np.int64]
        empty_nn_distances, empty_nn_indexes = _nearest_neighbor_query(
            normalized_scene_empty_point_cloud, basis_point_cloud
        )

        # if basis point inside object, replace with negative distance to nearest empty point
        # heuristic for "inside object": closer to a scene point than to an empty point
        for i in range(len(nn_distances)):
            if nn_distances[i] < empty_nn_distances[i]:
                nn_distances[i] = -empty_nn_distances[i]

    if encoding_type == "scalar":
        # return the nearest neighbor distances for scalar bps encoding
        if grid_shape_for_grid_basis is not None:
            # reshape bps encoding to a grid shape (for cnn models) if provided
            return nn_distances.reshape(grid_shape_for_grid_basis, order="F")
        return nn_distances
    elif encoding_type == "diff":
        # calculate and return difference vectors for diff bps encoding
        return _calculate_diff_vectors(normalized_scene_point_cloud, basis_point_cloud, nn_indexes)
    elif encoding_type == "both":
        # return both encoding types if requested
        return (
            nn_distances,
            _calculate_diff_vectors(normalized_scene_point_cloud, basis_point_cloud, nn_indexes),
        )
    else:
        raise ValueError("encoding should be one of 'scalar', 'diff', or 'both'")


"""
HELPER FUNCTIONS
"""


def _get_scene_normalization_params(
    scene_point_cloud: NDArray[np.float64], bound_shape: str
) -> tuple[NDArray[np.float64], np.float64]:
    """
    Get the centroid and maximum distance from the centroid of a scene.
    These parameters are necessary to normalize it.

    Parameters
    ----------
    scene_point_cloud: np.ndarray
        2d or 3d opints in scene
    bound_shape: str "ncube" or "nsphere"
        which shape to use for scene normalization
        affects how max distance is calculated

    Returns
    -------
    centroid: np.ndarray
        Average point coordinates
    max_distance_from_centroid: np.float64
        Maximum distance of any point in the cloud from the centroid
    """
    centroid: NDArray[np.float64] = cast(NDArray[np.float64], np.mean(scene_point_cloud, axis=0))
    centered_cloud: NDArray[np.float64] = scene_point_cloud - centroid

    max_distance: np.float64
    if bound_shape == "nsphere":
        distances_from_centroid: NDArray[np.float64] = cast(NDArray[np.float64], np.linalg.norm(centered_cloud, axis=1))
        max_distance = np.max(distances_from_centroid)
    elif bound_shape == "ncube":
        max_distance = cast(np.float64, np.max(np.abs(centered_cloud)))
    else:
        raise ValueError("bound_shape must be 'nsphere' or 'ncube'")

    return centroid, max_distance


def _normalize_scene(
    scene_point_cloud: NDArray[np.float64], bound_shape: str, scene_empty_point_cloud: None | NDArray[np.float64] = None
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Normalize the scene by centering points and scaling to fit within a unit n-sphere or unit n-cube.
    If a scene_empty_point_cloud is also provided, it will be normalized according to the parameters of the scene point cloud.

    Parameters
    ----------
    scene_point_cloud: np.ndarray
        2d or 3d points in scene
    bound_shape: str "ncube" or "nsphere" or "none"
        which shape to use for scene normalization
        use ncube with grid-based BPS or nsphere with random sampling-based BPS
        do not use "none" for ML-based scene reconstruction, only for procedural
    scene_empty_point_cloud: np.ndarray (optional)
        an empty point cloud which will also be normalized
        it will be normalized using centroid and max distance of scene_point_cloud, not its own

    Returns
    -------
    scene_normalized_point_cloud: np.ndarray
        2d or 3d points in normalized scene
    scene_normalized_empty_point_cloud: np.ndarray (optional)
        2d or 3d points in normalized empty point cloud from scene
    """
    if bound_shape == "none":
        if scene_empty_point_cloud is not None:
            return scene_point_cloud, scene_empty_point_cloud
        return scene_point_cloud

    centroid: NDArray[np.float64]
    max_distance: np.float64
    centroid, max_distance = _get_scene_normalization_params(scene_point_cloud, bound_shape)

    normalized_scene_point_cloud: NDArray[np.float64] = (scene_point_cloud - centroid) / max_distance

    if scene_empty_point_cloud is not None:
        normalized_scene_empty_point_cloud: NDArray[np.float64] = (scene_empty_point_cloud - centroid) / max_distance
        return normalized_scene_point_cloud, normalized_scene_empty_point_cloud

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
    scene_ball_tree: BallTree = BallTree(scene_point_cloud, leaf_size=40, metric="minkowski")
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
    x_norms = cast(NDArray[np.float64], np.sqrt(np.sum(np.square(x), axis=1)).reshape([-1, 1]))
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
