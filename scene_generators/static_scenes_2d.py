"""
Static Two-Dimensional Scene Generators
for ADLR Trajectory Planning Project
Moritz Schüler and Alexander João Peterson Santos
2025-11-04
"""

import numpy as np
import random
from typing import Tuple, List, Deque, Set, Dict
from collections import deque
from numpy import ndarray

class StaticScene2D:
    occupancy_grid: ndarray
    scene_concept: str
    dims: Tuple[int, int]
    start_point: Tuple[int, int]
    goal_point: Tuple[int, int]

    def __init__(self) -> None:
        pass

    def occupancy_grid(self) -> ndarray:
        return self.occupancy_grid

    def flat(self) -> ndarray:
        return self.occupancy_grid.flatten()

    def to_file(self, filename: str) -> None:
        """
        Write the scene to a file of given filename.
        ~~ outline the scene (not part of it)
        [] represents a point in the scene
        """
        f = open(filename, "w", encoding="utf8")
        f.write(f"{''.join(['~'] * (self.dims[1]*2 + 4))}\n" )
        f.write(f"{''.join(['~'] * (self.dims[1]*2 + 4))}\n" )
        for row in self.occupancy_grid:
            f.write("~~")
            for value in row:
                f.write("[]" if value != 0 else "  ")
            f.write("~~\n")
        f.write(f"{''.join(['~'] * (self.dims[1]*2 + 4))}\n" )
        f.write(f"{''.join(['~'] * (self.dims[1]*2 + 4))}\n" )
        f.close()


class StaticMaze2D(StaticScene2D):
    
    def __init__(
        self,
        dims: Tuple[int, int],  # height, width
        cell_length: int,  # wall length of each square-shaped chamber
        cycle_rate: float = 0.0,  # rate at which random walls are removed, forming cycles
        blocked_cell_rate: float = 0.0  # rate at which random chambers are filled in completely
    ) -> None:
        self.scene_concept = "maze"
        self.dims = dims
        maze = self.generate_maze(dims, cell_length, cycle_rate, blocked_cell_rate)
        self.start_point, self.goal_point, self.occupancy_grid = maze

    def generate_maze(
        self,
        dims: Tuple[int, int],  # (height, width) of the final array
        cell_length: int,  # The full length of one cell unit (path + 1 wall)
        cycle_rate: float = 0.0,  # rate at which random walls are removed, forming cycles
        blocked_cell_rate: float = 0.0,  # rate at which random chambers are filled in completely
    ) -> Tuple[Tuple[int, int], Tuple[int, int], ndarray]:
        """
        Generates a grid-based maze scene with square-shaped cells divided by walls.
        Returns starting and goal point along with occupancy grid representation of scene.
        """
        
        height, width = dims
        cell_inner_length: int = cell_length - 1  # cell length minus 1 wall block

        if cell_inner_length < 1:
            raise ValueError("cell_length must be at least 2 (1 free block + 1 wall block)")

        # calculate num rows/cols possible (we may need to pad the maze later to fit in given dims)
        num_rows: int = (height - 1) // cell_length
        num_cols: int = (width - 1) // cell_length
        maze_core_height: int = 1 + num_rows * cell_length
        maze_core_width: int = 1 + num_cols * cell_length
        if num_rows <= 0 or num_cols <= 0:
            raise ValueError("Dimensions are too small for the given cell_length.")

        # visited grid helps DFS algorithm generate maze by connecting unvisited cells to each other
        maze_core: ndarray = np.ones((maze_core_height, maze_core_width), dtype=np.uint8)
        visited: ndarray = np.zeros((num_rows, num_cols), dtype=bool)

        # choose random start and end cells
        all_cells: List[Tuple[int, int]] = [(r, c) for r in range(num_rows) for c in range(num_cols)]
        start_cell: Tuple[int, int]
        end_cell: Tuple[int, int]
        start_cell, end_cell = random.sample(all_cells, 2)
        start_r, start_c = start_cell
        end_r, end_c = end_cell
        
        def _get_cell_coords(r: int, c: int) -> Tuple[int, int]:
            """Returns the array coords for the top-left block of a cell"""
            y: int = 1 + r * cell_length
            x: int = 1 + c * cell_length
            return y, x
            
        def _get_center_coords(r: int, c: int) -> Tuple[int, int]:
            """Returns the array coords for the center of the cell's path area."""
            y_tl, x_tl = _get_cell_coords(r, c)
            offset: int = cell_inner_length // 2
            y_center: int = y_tl + offset
            x_center: int = x_tl + offset
            return y_center, x_center

        def _generate_core_maze(maze: ndarray) -> ndarray:
            """Create maze by connecting unvisited cells using DFS"""
            
            stack: Deque[Tuple[int, int]] = deque([start_cell])
            visited[start_r, start_c] = True

            start_y, start_x = _get_cell_coords(start_r, start_c)

            # free blocks in starting cell
            maze[start_y : start_y + cell_inner_length, start_x : start_x + cell_inner_length] = 0

            while stack:
                current_r, current_c = stack[-1]
                
                # randomly pick which cell to explore next from the current cell in DFS (this is how the maze is random)
                possible_moves: List[Tuple[int, int, str]] = [(0, 1, 'R'), (0, -1, 'L'), (1, 0, 'D'), (-1, 0, 'U')] 
                random.shuffle(possible_moves)
                
                neighbors: List[Tuple[int, int, int, int, str]] = []
                for dr, dc, direction in possible_moves:
                    next_r, next_c = current_r + dr, current_c + dc
                    
                    if 0 <= next_r < num_rows and 0 <= next_c < num_cols and not visited[next_r, next_c]:
                        neighbors.append((next_r, next_c, dr, dc, direction))

                if neighbors:
                    next_r, next_c, dr, dc, direction = neighbors[0]

                    # free blocks for new unvisited cell
                    next_y, next_x = _get_cell_coords(next_r, next_c)
                    maze[next_y : next_y + cell_inner_length, next_x : next_x + cell_inner_length] = 0

                    # free blocks between new cell and current cell in DFS
                    curr_y, curr_x = _get_cell_coords(current_r, current_c)
                    if direction == 'R':
                        wall_y, wall_x = curr_y, curr_x + cell_inner_length
                        maze[wall_y : wall_y + cell_inner_length, wall_x] = 0
                    elif direction == 'L':
                        wall_y, wall_x = next_y, next_x + cell_inner_length
                        maze[wall_y : wall_y + cell_inner_length, wall_x] = 0
                    elif direction == 'D':
                        wall_y, wall_x = curr_y + cell_inner_length, curr_x
                        maze[wall_y, wall_x : wall_x + cell_inner_length] = 0
                    elif direction == 'U':
                        wall_y, wall_x = next_y + cell_inner_length, next_x
                        maze[wall_y, wall_x : wall_x + cell_inner_length] = 0
                    
                    visited[next_r, next_c] = True
                    stack.append((next_r, next_c))
                else:
                    stack.pop()
            return maze

        def _break_walls(maze: ndarray, rate: float) -> ndarray:
            """Randomly converts internal walls to path based on rate (cycle_rate)."""
            if rate <= 0.0:
                return maze
            
            # break horizontal walls
            for r in range(num_rows - 1):
                wall_y: int = 1 + (r + 1) * cell_length - 1
                for c in range(num_cols):
                    wall_x_start: int = 1 + c * cell_length
                    wall_x_end: int = wall_x_start + cell_inner_length
                    
                    if maze[wall_y, wall_x_start] == 1 and random.random() < rate:
                        maze[wall_y, wall_x_start : wall_x_end] = 0

            # break vertical walls
            for c in range(num_cols - 1):
                wall_x: int = 1 + (c + 1) * cell_length - 1
                for r in range(num_rows):
                    wall_y_start: int = 1 + r * cell_length
                    wall_y_end: int = wall_y_start + cell_inner_length
                    
                    if maze[wall_y_start, wall_x] == 1 and random.random() < rate:
                        maze[wall_y_start : wall_y_end, wall_x] = 0
            return maze

        def _remove_isolated_wall_corners(maze: ndarray) -> ndarray:
            """Removes wall blocks with 0 wall neighbors (isolated blocks)."""
            temp_maze: ndarray = maze.copy()
            
            for r in range(1, maze_core_height - 1):
                for c in range(1, maze_core_width - 1):
                    
                    if maze[r, c] == 1:
                        neighbor_count: int = 0
                        
                        # 4-connectivity check (N, S, E, W)
                        if maze[r - 1, c] == 1: neighbor_count += 1
                        if maze[r + 1, c] == 1: neighbor_count += 1
                        if maze[r, c - 1] == 1: neighbor_count += 1
                        if maze[r, c + 1] == 1: neighbor_count += 1
                            
                        if neighbor_count == 0:
                            temp_maze[r, c] = 0 
            
            return temp_maze
        
        def _build_adjacency_list(maze: ndarray) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
            """
            Build adjacency lift for which cells are connected to each other.
            Useful for identifying cells not along the path from start-> goal cell, so we can fill them if requested.
            """
            adj_list: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
            
            for r, c in all_cells:
                adj_list[(r, c)] = []
                
                # check 4 neighbors
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    
                    if 0 <= nr < num_rows and 0 <= nc < num_cols:
                        # determine wall location in the maze array between (r, c) and (nr, nc)
                        wall_y: int; wall_x: int
                        if dr == 1: wall_y, wall_x = _get_cell_coords(r, c)[0] + cell_inner_length, _get_cell_coords(r, c)[1]
                        elif dr == -1: wall_y, wall_x = _get_cell_coords(nr, nc)[0] + cell_inner_length, _get_cell_coords(nr, nc)[1]
                        elif dc == 1: wall_y, wall_x = _get_cell_coords(r, c)[0], _get_cell_coords(r, c)[1] + cell_inner_length
                        elif dc == -1: wall_y, wall_x = _get_cell_coords(nr, nc)[0], _get_cell_coords(nr, nc)[1] + cell_inner_length
                        else: continue 

                        is_passage_open: bool = False
                        if dr != 0: # horizontal wall
                            if np.any(maze[wall_y, wall_x : wall_x + cell_inner_length] == 0): is_passage_open = True
                        else: # vertical wall
                            if np.any(maze[wall_y : wall_y + cell_inner_length, wall_x] == 0): is_passage_open = True

                        if is_passage_open:
                            adj_list[(r, c)].append((nr, nc))
                            
            return adj_list

        def _find_path_and_identify_fillable_cells(adj_list: Dict[Tuple[int, int], List[Tuple[int, int]]]) -> Set[Tuple[int, int]]:
            """
            Uses BFS on the adjacency list to find the shortest path
            Identifies cells not on that specific path as fillable (available to block if desired).
            """
            # Item in queue: (r, c, path_list_to_here)
            queue: Deque[Tuple[int, int, List[Tuple[int, int]]]] = deque([(start_r, start_c, [start_cell])])
            visited_bfs: Set[Tuple[int, int]] = {start_cell}
            
            shortest_path_cells: Set[Tuple[int, int]] = set()

            while queue:
                r, c, path = queue.popleft()
                
                if (r, c) == end_cell:
                    # find cells which are not fillable
                    shortest_path_cells = set(path)
                    break 

                for nr, nc in adj_list.get((r, c), []):
                    if (nr, nc) not in visited_bfs:
                        visited_bfs.add((nr, nc))
                        queue.append((nr, nc, path + [(nr, nc)]))
            
            # ensure start/end are not filled.
            if not shortest_path_cells:
                 shortest_path_cells = {start_cell, end_cell}

            # cells not on the shortest path are available to be filled
            fillable_cells: Set[Tuple[int, int]] = set(all_cells) - shortest_path_cells
            return fillable_cells

        def _fill_random_cells(maze: ndarray, fillable_cells: Set[Tuple[int, int]], rate: float) -> ndarray:
            """Fills a proportion of fillable cells completely with walls (1s)."""
            if rate <= 0.0 or not fillable_cells:
                return maze

            # choose which ones to fill based on given proportion
            num_to_fill: int = int(len(fillable_cells) * rate)
            cells_to_fill: List[Tuple[int, int]] = random.sample(list(fillable_cells), k=num_to_fill)
            
            for r, c in cells_to_fill:
                y, x = _get_cell_coords(r, c)
                maze[y-1 : y + cell_length, x-1 : x + cell_length] = 1
                
            return maze

        
        # generate maze with path from start_cell to end_cell
        # maze has no cycles and no filled cells
        maze_core = _generate_core_maze(maze_core)
        
        # introduce cycles if requested
        if cycle_rate > 0.0:
            maze_core = _break_walls(maze_core, cycle_rate)
            
        # block out some cells if requested
        if blocked_cell_rate > 0:
            adj_list = _build_adjacency_list(maze_core)
            fillable_cells = _find_path_and_identify_fillable_cells(adj_list)
            maze_core = _fill_random_cells(maze_core, fillable_cells, blocked_cell_rate)

        # remove wall corners not connected to any walls
        maze_core = _remove_isolated_wall_corners(maze_core)

        # padding (in case maze is smaller than desired scene dims)
        final_maze: ndarray = np.ones((height, width), dtype=np.uint8)
        final_maze[0:maze_core_height, 0:maze_core_width] = maze_core
        
        # get coordinates for start and goal cells
        start_center_coords: Tuple[int, int] = _get_center_coords(start_r, start_c)
        end_center_coords: Tuple[int, int] = _get_center_coords(end_r, end_c)
        
        return start_center_coords, end_center_coords, final_maze
 