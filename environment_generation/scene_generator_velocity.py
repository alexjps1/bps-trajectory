"""
Script for Generating Dynamic 2D Scenes with Velocity Information
for ADLR Trajectory Planning Project
Moritz Schüler and Alexander João Peterson Santos
2025-01-07

Adapted from environment_generation.ipynb with some changes,
including typing and velocity vectors for trajectories.

Scenes are stored as np.float32 to accomodate velocities, which can be non-integer,
and to facilitate ML training, as our models typically use float32 inputs.

Note that cells with multiple (overlapping) objects will have the velocity of the last-painted object.
There is no blending of velocities or special magic value to indicate multiple objects in one cell.
"""

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from numpy.typing import NDArray


def compute_velocity_vectors(trajectory: NDArray[np.float32], use_unitvecs: bool) -> NDArray[np.float32]:
    """
    Compute velocity vectors for a trajectory.

    The velocity vector for each frame represents the motion from the previous
    frame to the current frame (delta_y, delta_x). The first frame has a
    velocity of (0, 0) since there is no previous frame.

    Parameters
    ----------
    trajectory : NDArray[np.float32]
        Trajectory positions with shape (frames, 2), where each row is [y, x].
    use_unitvecs : bool
        If True, normalize velocity vectors to unit vectors (removing speed information).
        If False, velocity vectors retain their original magnitude.

    Returns
    -------
    NDArray[np.float32]
        Velocity vectors with shape (frames, 2), where each row is [delta_y, delta_x].
        The first row is always [0, 0].
    """
    assert trajectory.ndim == 2 and trajectory.shape[1] == 2, "Trajectory must have shape (frames, 2)"
    num_frames = trajectory.shape[0]
    velocities = np.zeros((num_frames, 2), dtype=np.float32)

    if num_frames <= 1:
        # one frame means no velocities (just leave the array with all zeros)
        return velocities

    # Compute deltas: current position - previous position
    velocities[1:] = trajectory[1:].astype(np.float32) - trajectory[:-1].astype(np.float32)

    if not use_unitvecs:
        return velocities

    # normalize to unit vectors if requested
    magnitudes = np.linalg.norm(velocities, axis=1, keepdims=True)
    velocities = np.where(magnitudes != 0, velocities / magnitudes, 0)

    return velocities


def plot_environment(env: NDArray[np.float32]) -> None:
    """
    Display an environment grid as a binary image.

    Parameters
    ----------
    env : NDArray[np.float32]
        2D array representing the environment, where 0 is free space and 1 is occupied.

    Returns
    -------
    None
    """
    plt.imshow(env, cmap="binary")
    plt.show()


def get_circle(radius: int, center: tuple[int, int], env_size: tuple[int, int]) -> NDArray[np.float32]:
    """
    Generate a binary mask of a filled circle.

    Parameters
    ----------
    radius : int
        Radius of the circle in pixels.
    center : tuple[int, int]
        Center coordinates (y, x) of the circle.
    env_size : tuple[int, int]
        Size of the environment grid (height, width).

    Returns
    -------
    NDArray[np.float32]
        Binary mask of shape env_size with 1s inside the circle and 0s outside.
    """
    layer = np.zeros(env_size).astype(np.float32)
    for y in range(env_size[0]):
        for x in range(env_size[1]):
            if (y - center[0]) ** 2 + (x - center[1]) ** 2 <= radius**2:
                layer[y, x] = 1
    return layer.astype(np.float32)


def get_star(
    center: tuple[int, int], env_size: tuple[int, int], num_points: int, outer_radius: int, inner_radius: int
) -> NDArray[np.float32]:
    """
    Generate a binary mask of a filled star shape.

    Parameters
    ----------
    center : tuple[int, int]
        Center coordinates (y, x) of the star.
    env_size : tuple[int, int]
        Size of the environment grid (height, width).
    num_points : int
        Number of points (tips) on the star.
    outer_radius : int
        Distance from center to the outer points (tips) of the star.
    inner_radius : int
        Distance from center to the inner vertices between points.

    Returns
    -------
    NDArray[np.float32]
        Binary mask of shape env_size with 1s inside the star and 0s outside.
    """
    size = env_size[0]

    points = []
    for i in range(num_points * 2):
        angle = np.pi / num_points * i - np.pi / 2
        r = outer_radius if i % 2 == 0 else inner_radius
        x = center[0] + r * np.cos(angle)
        y = center[1] + r * np.sin(angle)
        points.append((x, y))

    star_path = Path(points)

    y, x = np.mgrid[0:size, 0:size]
    coords = np.vstack((x.ravel(), y.ravel())).T

    mask = star_path.contains_points(coords)
    star_array = mask.reshape((size, size))

    return star_array.astype(np.float32)


def get_circular_trajectory(
    start: Tuple[int, int],
    radius: int,
    speed: float,
    clock_wise: bool,
    compute_velocity: bool,
    velocity_unitvec: bool = False,
) -> Tuple[NDArray[np.float32], NDArray[np.float32] | None]:
    """
    Get a circular trajectory for an object and optionally the corresponding velocity vectors.

    Parameters
    ----------
    start : Tuple[int, int]
        Starting coordinate (y, x) of the trajectory.
    radius : int
        Radius of the circular trajectory.
    speed : float
        Speed of the object (higher values = fewer points in trajectory).
    clock_wise : bool
        If True, object moves clockwise; if False, counter-clockwise.
    compute_velocity : bool
        Whether to compute and return velocity vectors.
    velocity_unitvec : bool, optional
        If True, normalize velocity vectors to unit vectors. Defaults to False.

    Returns
    -------
    Tuple[NDArray[np.float32], NDArray[np.float32] | None]
        trajectory : NDArray[np.float32]
            Array of points on the circular trajectory with shape (n_points, 2).
        velocities : NDArray[np.float32] | None
            Velocity vectors with shape (n_points, 2) if compute_velocity=True, else None.
    """
    trajectory: List[NDArray[np.float32]] = []
    x, y = start
    x_0, y_0 = start - np.array([radius, 0]).astype(np.float32)
    angle_0 = np.arctan2(y - y_0, x - x_0).astype(np.float32)

    linspace = (
        np.linspace(angle_0, angle_0 + 2 * np.pi, int(100 / speed))
        if clock_wise
        else reversed(np.linspace(angle_0, angle_0 + 2 * np.pi, int(100 / speed)))
    )
    for angle in linspace:
        x = np.float32(np.around(x_0 + radius * np.cos(angle)))
        y = np.float32(np.around(y_0 + radius * np.sin(angle)))

        point = np.array([x, y])
        trajectory.append(point)

    trajectory_arr: NDArray[np.float32] = np.stack(trajectory, axis=0).astype(np.float32)
    velocity_arr: NDArray[np.float32] | None = (
        compute_velocity_vectors(trajectory_arr, velocity_unitvec) if compute_velocity else None
    )

    return (trajectory_arr, velocity_arr)


def get_linear_trajectory(
    start: tuple[int, int], end: tuple[int, int], speed: float, compute_velocity: bool, velocity_unitvec: bool = False
) -> Tuple[NDArray[np.float32], NDArray[np.float32] | None]:
    """
    Get a linear trajectory for an object and optionally the corresponding velocity vectors.

    Parameters
    ----------
    start : tuple[int, int]
        Starting coordinate (y, x) of the trajectory.
    end : tuple[int, int]
        Ending coordinate (y, x) of the trajectory.
    speed : float
        Speed of the object (higher values = fewer points in trajectory).
    compute_velocity : bool
        Whether to compute and return velocity vectors.
    velocity_unitvec : bool, optional
        If True, normalize velocity vectors to unit vectors. Defaults to False.

    Returns
    -------
    Tuple[NDArray[np.float32], NDArray[np.float32] | None]
        trajectory : NDArray[np.float32]
            Array of points on the linear trajectory with shape (n_points, 2).
        velocities : NDArray[np.float32] | None
            Velocity vectors with shape (n_points, 2) if compute_velocity=True, else None.
    """
    trajectory: List[NDArray[np.float32]] = []
    startarr = np.array(start).astype(np.float32)
    endarr = np.array(end).astype(np.float32)
    vector = endarr - startarr
    norm = np.linalg.norm(vector) / speed
    vector_norm = vector / norm if norm > 0 else vector

    for i in range(1, int(norm)):
        x, y = startarr + i * vector_norm
        point = np.array([np.float32(np.around(x)), np.float32(np.around(y))])
        trajectory.append(point)

    trajectory.append(endarr)

    trajectory_arr: NDArray[np.float32] = np.stack(trajectory, axis=0).astype(np.float32)
    velocity_arr: NDArray[np.float32] | None = (
        compute_velocity_vectors(trajectory_arr, velocity_unitvec) if compute_velocity else None
    )

    return (trajectory_arr, velocity_arr)


def get_virtual_point(env_size: tuple[int, int], radius: int, virtual_points: NDArray[np.float32]) -> tuple[int, int]:
    """
    Get a random point outside the grid at a distance ensuring the object fully exits.

    This point is used as a start or end point for linear trajectories, ensuring
    objects spawn from or disappear to positions outside the visible environment.

    Parameters
    ----------
    env_size : tuple[int, int]
        Size of the environment (height, width), e.g., (64, 64).
    radius : int
        Radius of the object moving on the trajectory.
    virtual_points : NDArray[np.float32]
        Array of coordinates on a virtual border around the environment (width 1).

    Returns
    -------
    tuple[int, int]
        Random point (y, x) outside the environment, offset by the object's radius
        to ensure the entire object is outside the grid boundary.
    """
    random_point = virtual_points[np.random.randint(len(virtual_points))]
    point_y, point_x = random_point
    # ensure that the whole object disappears, before reaching its destination
    if point_y == -1:
        point_y -= radius
    if point_y == env_size[0]:
        point_y += radius
    if point_x == -1:
        point_x -= radius
    if point_x == env_size[0]:
        point_x += radius
    point = point_y, point_x

    return point


def random_object_dynamic_env(
    env_size: tuple[int, int],
    n_objects: int,
    n_frames: int,
    encode_velocity: bool,
    velocity_unitvec: bool = False,
) -> NDArray[np.float32]:
    """
    Generate a random dynamic grid environment with moving objects.

    Creates an environment with randomly placed circles and stars that can move
    along linear or circular trajectories. Objects can optionally encode velocity
    information in additional channels.

    Parameters
    ----------
    env_size : tuple[int, int]
        Size of the environment (height, width), e.g., (64, 64).
    n_objects : int
        Number of objects in the environment.
    n_frames : int
        Number of frames of the dynamic environment. Use n_frames=1 for static environments.
    encode_velocity : bool
        If True, output includes velocity channels (shape: [frames, h, w, 3]).
        If False, output is occupancy only (shape: [frames, h, w]).
    velocity_unitvec : bool, optional
        If True, normalize velocity vectors to unit vectors (removing speed information).
        Defaults to False.

    Returns
    -------
    NDArray[np.float32]
        Array of frames representing the dynamic environment.
        Shape is (n_frames, height, width, 3) if encode_velocity=True,
        or (n_frames, height, width) if encode_velocity=False.
        Channel 0 contains occupancy (0=free, 1=occupied).
        Channels 1-2 contain velocity (delta_y, delta_x) when encode_velocity=True.
    """

    def initialize_objects(frames_arr: NDArray[np.float32], virtual_points: NDArray[np.float32]) -> Dict[int, Dict]:
        """
        Create an object_dict entry for new objects, containing randomly-chosen attributes, such as shape, trajectory, and speed.
        Does not paint the object.
        """
        # initialization of random objects
        initial_object_dict: Dict[int, Dict] = {}

        for id in range(n_objects):
            center_y, center_x = np.random.choice(np.arange(env_size[0])), np.random.choice(np.arange(env_size[0]))
            # object is a circle with probability of 50%
            if np.random.random() > 0.5:
                radius = np.random.choice(np.arange(10))
                obj_mask = get_circle(radius, (center_y, center_x), env_size)
                object_meta_data = radius
            # object is a star with probability of 50%
            else:
                num_points = np.random.choice(np.arange(3, 7))
                radius = np.random.choice(np.arange(5, 8))
                inner_radius = np.random.choice(np.arange(1, 5))
                obj_mask = get_star((center_y, center_x), env_size, num_points, radius, inner_radius)
                object_meta_data = [num_points, radius, inner_radius]

            # object is moving with a probability of 50%
            if np.random.random() > 0.5 and n_frames > 1:
                start = center_y, center_x
                speed = np.random.choice(np.arange(0.25, 1, 0.25))
                # object is moving on a linear trajectory with a probability of 70%
                if np.random.random() > 0.3:
                    end = get_virtual_point(env_size, radius, virtual_points)
                    trajectory, velocities = get_linear_trajectory(start, end, speed, encode_velocity, velocity_unitvec)
                    trajectory_type = "linear"
                # object is moving on a circular trajectory with a probability of 30%
                else:
                    radius = np.random.choice(np.arange(env_size[0] / 2, env_size[0]))
                    # the circular trajectory is clockwise with a probability of 50%
                    if np.random.random() > 0.5:
                        clock_wise = True
                    else:
                        clock_wise = False
                    trajectory, velocities = get_circular_trajectory(
                        start, radius, speed, clock_wise, encode_velocity, velocity_unitvec
                    )
                    trajectory_type = "circular"
            else:
                # object is static
                trajectory = None
                trajectory_type = None
                velocities = None
                speed = None

            assert obj_mask.shape == env_size
            # add dimensions for velocity if needed (velocity is 0 for initialized objects)
            if encode_velocity:
                obj_occupancy = np.zeros((env_size[0], env_size[1], 3), dtype=np.float32)
                obj_occupancy[:, :, 0] = obj_mask
            else:
                obj_occupancy = obj_mask

            # Store trajectory, velocities, index (0) for tracking position, and speed
            initial_object_dict[id] = {
                "obj_occupancy": obj_occupancy,
                "object_meta_data": object_meta_data,
                "trajectory": trajectory,
                "trajectory_type": trajectory_type,
                "velocities": velocities,
                "traj_idx": 0,
                "speed": speed,
            }

            # DEBUG, this block can be removed
            if encode_velocity:
                assert initial_object_dict[id]["obj_occupancy"].shape == (env_size[0], env_size[1], 3)
            else:
                assert initial_object_dict[id]["obj_occupancy"].shape == env_size

        return initial_object_dict

    def reinitialize_linear_trajectory_object() -> Dict:
        """
        Return list of randomly-chosen attributes for a new object with linear trajectory.
        This function is intended to be used when a moving object with a linear trajectory exhausts its trajectory list.
        Does not paint the object.
        """
        new_center_y, new_center_x = (
            np.random.choice(np.arange(env_size[0])),
            np.random.choice(np.arange(env_size[0])),
        )
        new_speed = np.random.choice(np.arange(0.25, 1, 0.25))
        if np.random.random() > 0.5:
            radius = np.random.choice(np.arange(10))
            obj_occupancy = get_circle(radius, (new_center_y, new_center_x), env_size)
            object_meta_data = radius
        else:
            num_points = np.random.choice(np.arange(3, 7))
            radius = np.random.choice(np.arange(5, 8))
            inner_radius = np.random.choice(np.arange(1, 5))
            obj_occupancy = get_star((new_center_y, new_center_x), env_size, num_points, radius, inner_radius)
            object_meta_data = [num_points, radius, inner_radius]

        start = get_virtual_point(env_size, radius, virtual_points)
        end = get_virtual_point(env_size, radius, virtual_points)
        trajectory, velocities = get_linear_trajectory(start, end, new_speed, encode_velocity, velocity_unitvec)
        trajectory_type = "linear"

        # Save the occupancy of the first frame of this object in the dict
        next_point = trajectory[0]
        if isinstance(object_meta_data, list):
            num_points, radius, inner_radius = object_meta_data
            obj_mask = get_star(tuple(next_point), env_size, num_points, radius, inner_radius)
        else:
            radius = object_meta_data
            obj_mask = get_circle(radius, tuple(next_point), env_size)

        # save occupancy (and ggf velocities) of first frame of this object in the dict
        assert obj_mask.shape == env_size
        # add dimensions for velocity if needed (velocity is 0 for initialized objects)
        if encode_velocity:
            obj_occupancy = np.zeros((env_size[0], env_size[1], 3), dtype=np.float32)
            obj_occupancy[:, :, 0] = obj_mask
        else:
            obj_occupancy = obj_mask

        return {
            "obj_occupancy": obj_occupancy,
            "object_meta_data": object_meta_data,
            "trajectory": trajectory,
            "trajectory_type": trajectory_type,
            "velocities": velocities,
            "traj_idx": 0,
            "speed": new_speed,
        }

    def iterate_objects(object_dict: Dict[int, Dict]) -> Dict[int, Dict]:
        """
        Update the entry of each object in the object_dict to reflect its next frame.
        Returns the updated dict. Does not paint objects onto the frame_arr.
        """
        for id, object_data in object_dict.items():
            object_meta_data = object_data["object_meta_data"]
            trajectory = object_data["trajectory"]
            trajectory_type = object_data["trajectory_type"]
            velocities = object_data["velocities"]
            traj_idx = object_data["traj_idx"]

            if trajectory is None:
                # static object need not be updated
                continue

            if traj_idx >= len(trajectory):
                # trajectory exhausted, which only happens with linear trajectories
                # replace its id with a new object entry which also has linear trajectory
                object_dict[id] = reinitialize_linear_trajectory_object()

            # the object is placed on its next point based on its trajectory
            else:
                next_point = trajectory[traj_idx]
                new_occupancy: NDArray[np.float32]
                if encode_velocity:
                    new_occupancy = np.zeros((env_size[0], env_size[1], 3)).astype(np.float32)
                else:
                    new_occupancy = np.zeros(env_size).astype(np.float32)

                if isinstance(object_meta_data, list):
                    num_points, radius, inner_radius = object_meta_data
                    obj_mask = get_star(tuple(next_point), env_size, num_points, radius, inner_radius)
                else:
                    radius = object_meta_data
                    obj_mask = get_circle(radius, tuple(next_point), env_size)

                if encode_velocity:
                    new_occupancy[:, :, 0] += obj_mask
                    next_velocity = velocities[traj_idx]
                    new_occupancy[:, :, 1][obj_mask != 0] = next_velocity[0]
                    new_occupancy[:, :, 2][obj_mask != 0] = next_velocity[1]
                else:
                    new_occupancy[:, :] += obj_mask

                # increment index for next frame, but wrapping circular trajectories which must never be exhausted
                updated_idx = (traj_idx + 1) % len(trajectory) if trajectory_type == "circular" else traj_idx + 1

                # update entry for this object
                object_dict[id]["obj_occupancy"] = new_occupancy
                object_dict[id]["traj_idx"] = updated_idx

        return object_dict

    def paint_objects(frames_arr: NDArray[np.float32], object_dict: Dict[int, Dict], frame_num: int) -> None:
        """
        Modify the current frame by reference by painting each object form the object_dict onto it.
        This will not iterate the trajectory indices or make any modifications whatsoever to the object_dict.
        Assumes that obj_occupancy and frames_arr have shapes in accordance with the value of encode_velocity.
        """
        if encode_velocity:
            for _, obj_data in object_dict.items():
                frames_arr[frame_num, :, :, :] += obj_data["obj_occupancy"]
        else:
            for _, obj_data in object_dict.items():
                frames_arr[frame_num, :, :] += obj_data["obj_occupancy"]

    assert n_frames > 0

    if encode_velocity:
        # shape is [frames, rows, cols, 3] where the last dim contains (occupancy_val, delta_rows, delta_cols)
        frames_arr: NDArray[np.float32] = np.zeros((n_frames, env_size[0], env_size[1], 3)).astype(np.float32)
    else:
        # shape is [frames, rows, cols]
        frames_arr = np.zeros((n_frames, env_size[0], env_size[1])).astype(np.float32)

    # points around the environment
    virtual_environment = np.ones(np.array(env_size) + 2).astype(
        np.float32
    )  # virtual environment that is 1 pixel larger on each side
    virtual_environment[1 : env_size[0] + 1, 1 : env_size[0] + 1] = 0
    indices = np.where(virtual_environment == 1)
    virtual_points = (np.array(list(zip(*indices))) - 1).astype(
        np.float32
    )  # points that only exist in the virtual environment

    # initialize objects on first frame, painting them onto frames_arr
    object_dict: Dict[int, Dict] = initialize_objects(frames_arr, virtual_points)

    # DEBUG, this block can be removed later if desired
    assert len(object_dict) >= 1
    shape = object_dict[0]["obj_occupancy"].shape
    if encode_velocity:
        assert len(shape) == 3
        assert shape == (env_size[0], env_size[1], 3)
    else:
        assert len(shape) == 2
        assert shape == env_size

    # paint each object onto first frame
    paint_objects(frames_arr, object_dict, frame_num=0)

    if n_frames == 1:
        return frames_arr

    # iteratively update objects and paint frames for dynamic environments
    for frame_num in range(1, n_frames):
        object_dict = iterate_objects(object_dict)
        paint_objects(frames_arr, object_dict, frame_num)

    # force nonzero occupancy values to 1
    if encode_velocity:
        frames_arr[..., 0] = (frames_arr[..., 0] != 0).astype(np.float32)
    else:
        frames_arr = (frames_arr != 0).astype(np.float32)

    return frames_arr


def generate_training_data(
    env_size: tuple[int, int],
    n_objects: int,
    n_envs: int,
    n_frames: int = 1,
    iter: int = 0,
    encode_velocity: bool = False,
    velocity_unitvec: bool = False,
) -> None:
    """
    Generate and save training data for frame prediction models.

    Creates multiple dynamic scenes and saves them to a .npy file for use in
    training reconstruction loss or frame prediction models.

    Parameters
    ----------
    env_size : tuple[int, int]
        Size of each environment (height, width), e.g., (64, 64).
    n_objects : int
        Number of objects per environment.
    n_envs : int
        Number of environments (scenes) to generate.
    n_frames : int, optional
        Number of frames per environment. Defaults to 1.
    iter : int, optional
        Iteration number used in the output filename for batch generation. Defaults to 0.
    encode_velocity : bool, optional
        If True, include velocity channels in the output. Defaults to False.
    velocity_unitvec : bool, optional
        If True, normalize velocity vectors to unit vectors. Defaults to False.

    Returns
    -------
    None
        Saves the generated scenes to a .npy file in the dynamic_scenes2d directory.
        Filename format: dynamic_scenes2d_motionv01_{size}_{n_objects}_{n_envs}_{n_frames}_vel{0|1}_pt{iter:02}.npy
    """
    scenes_list = []
    for i in range(n_envs):
        print(f"Generating scene {i}/{n_envs} of file {iter}...")
        scene = random_object_dynamic_env(env_size, n_objects, n_frames, encode_velocity, velocity_unitvec)

        # DEBUG, this block can be removed later if desired
        assert isinstance(scene, np.ndarray)
        if encode_velocity:
            assert scene.shape == (n_frames, env_size[0], env_size[1], 3)
        else:
            assert scene.shape == (n_frames, env_size[0], env_size[1])
        scenes_list.append(scene)

    scenes_arr: NDArray[np.float32] = np.stack(scenes_list, axis=0).astype(np.float32)

    if encode_velocity:
        assert scenes_arr.shape == (n_envs, n_frames, env_size[0], env_size[1], 3)
        if iter == 0:
            print(
                f"The generated scenes include velocity information.\nNPY shape: (n_envs, n_frames, nrows, ncols, 3),  i.e. ({n_envs}, {n_frames}, {env_size[0]}, {env_size[1]}, 3)"
            )
    else:
        assert scenes_arr.shape == (n_envs, n_frames, env_size[0], env_size[1])
        if iter == 0:
            print(
                f"The generated scenes do NOT include velocity information.\nNPY shape: (n_envs, n_frames, nrows, ncols),  i.e. ({n_envs}, {n_frames}, {env_size[0]}, {env_size[1]})"
            )

    np.save(
        # assue cwd is the TrajectoryPlanning directory (project root)
        f"./frame_prediction/dynamic_scenes2d/dynamic_scenes2d_motionv01_{env_size[0]}_{n_objects}_{n_envs}_{n_frames}_vel{int(encode_velocity)}_pt{iter:02}.npy",
        scenes_arr,
    )


def main() -> None:
    """
    Generate a batch of training data files.

    Creates 10 .npy files containing dynamic scene data for training.
    Each file contains environments with 10 objects over 100 frames.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    num_files_to_generate = 10
    for iter in range(num_files_to_generate):
        print(f"Generating file {iter}... (generating {num_files_to_generate} files total)")
        generate_training_data(
            env_size=(64, 64),
            n_objects=10,
            n_envs=1000,
            n_frames=100,
            iter=iter,
            encode_velocity=True,
            velocity_unitvec=False,
        )
    print("Done generating files. Exiting...")


if __name__ == "__main__":
    main()
