"""Two-room gridworld: a point agent and a block that moves only when pushed.

Clean-room analogue of the limited-capacity object experiment in Ren et al.,
AGL-TR-2026-04, Section B, on a maze whose task distance is a shortest path
(Section 3, PointMaze geodesic). Not the paper's visual simulator.
"""

from __future__ import annotations

import numpy as np

# Stay, north, east, south, west.
DIRS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (-1, 0),
    (0, 1),
    (1, 0),
    (0, -1),
)
N_ACTIONS = 5


def make_maze(n: int) -> np.ndarray:
    """n×n two-room maze. n is odd and at least 7. Door is the center cell of the middle row."""
    if n < 7 or n % 2 == 0:
        raise ValueError(f"grid_size must be odd and >= 7, got {n}")
    grid = np.ones((n, n), dtype=np.int8)
    grid[1 : n - 1, 1 : n - 1] = 0
    mid = n // 2
    grid[mid, :] = 1
    grid[mid, mid] = 0
    return grid


def free_cells(maze: np.ndarray) -> list[tuple[int, int]]:
    cells = list(zip(*np.where(maze == 0), strict=True))
    return [(int(r), int(c)) for r, c in cells]


def geodesic_table(maze: np.ndarray) -> tuple[dict[tuple[int, int], int], np.ndarray]:
    """All-pairs shortest-path distances on the 4-connected free cells."""
    cells = free_cells(maze)
    index = {cell: i for i, cell in enumerate(cells)}
    n = len(cells)
    dist = np.full((n, n), np.inf, dtype=np.float64)
    neighbors = []
    for r, c in cells:
        nbrs = []
        for dr, dc in DIRS[1:]:
            rr, cc = r + dr, c + dc
            if (rr, cc) in index:
                nbrs.append(index[(rr, cc)])
        neighbors.append(nbrs)
    for start in range(n):
        dist[start, start] = 0.0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in neighbors[u]:
                if dist[start, v] == np.inf:
                    dist[start, v] = dist[start, u] + 1.0
                    queue.append(v)
    if not np.isfinite(dist).all():
        raise RuntimeError("maze free cells are not connected")
    return index, dist


def _bfs_path(
    maze: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    index, _ = geodesic_table(maze)
    if start not in index or goal not in index:
        raise ValueError("start or goal is a wall")
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    queue = [start]
    head = 0
    while head < len(queue):
        cur = queue[head]
        head += 1
        if cur == goal:
            break
        r, c = cur
        for dr, dc in DIRS[1:]:
            nxt = (r + dr, c + dc)
            if nxt in index and nxt not in prev:
                prev[nxt] = cur
                queue.append(nxt)
    if goal not in prev:
        raise RuntimeError("no path")
    path = [goal]
    while path[-1] != start:
        parent = prev[path[-1]]
        if parent is None:
            break
        path.append(parent)
    path.reverse()
    return path


def scripted_route(maze: np.ndarray) -> list[tuple[int, int]]:
    """Shortest path from the top-left free cell to the farthest free cell."""
    index, dist = geodesic_table(maze)
    start = (1, 1)
    start_i = index[start]
    far_i = int(np.argmax(dist[start_i]))
    cells = free_cells(maze)
    return _bfs_path(maze, start, cells[far_i])


def parked_block(maze: np.ndarray, path: list[tuple[int, int]]) -> tuple[int, int]:
    """A free cell off the scripted path, farthest from that path (the resting block)."""
    path_set = set(path)
    index, dist = geodesic_table(maze)
    best: tuple[int, int] | None = None
    best_d = -1.0
    for cell in free_cells(maze):
        if cell in path_set:
            continue
        d = min(dist[index[cell], index[p]] for p in path)
        if d > best_d:
            best_d = d
            best = cell
    if best is None:
        raise RuntimeError("no cell to park the block off the scripted path")
    return best


def _obs_from_positions(
    agent: np.ndarray,
    block: np.ndarray,
    n: int,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    t = agent.shape[0]
    obs = np.zeros((t, 2, n, n), dtype=np.float32)
    obs[np.arange(t), 0, agent[:, 0], agent[:, 1]] = 1.0
    obs[np.arange(t), 1, block[:, 0], block[:, 1]] = 1.0
    if noise_std > 0:
        obs += rng.normal(0.0, noise_std, size=obs.shape).astype(np.float32)
    return obs


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def rollout(
    maze: np.ndarray,
    *,
    length: int,
    seed: int,
    noise_std: float,
    contact_radius: int = 1,
) -> dict[str, np.ndarray]:
    """Uniform-random exploration on the maze.

    The block stays put until the agent enters its contact radius, then jumps
    to a random free cell outside that radius. This is the grid form of the
    contact-driven object in Ren et al. Section B ("the object jumps to a
    random position when the arm enters a contact radius").
    """
    if length < 8:
        raise ValueError("trajectory length must be at least 8")
    if contact_radius < 1:
        raise ValueError("contact_radius must be >= 1")
    n = int(maze.shape[0])
    cells = free_cells(maze)
    rng = np.random.default_rng(seed)
    agent_rc = (1, 1)
    block_rc = (n - 2, n // 2)
    if maze[agent_rc] != 0 or maze[block_rc] != 0:
        raise RuntimeError("start cells are walls")
    if _chebyshev(agent_rc, block_rc) <= contact_radius:
        raise RuntimeError("agent starts inside the block's contact radius")

    agent = np.empty((length, 2), dtype=np.int64)
    block = np.empty((length, 2), dtype=np.int64)
    actions = np.zeros(length, dtype=np.int64)
    contact = np.zeros(length, dtype=np.int64)
    was_inside = False

    for t in range(length):
        agent[t, 0], agent[t, 1] = agent_rc
        block[t, 0], block[t, 1] = block_rc
        if t == length - 1:
            break
        action = int(rng.integers(0, N_ACTIONS))
        actions[t] = action
        dr, dc = DIRS[action]
        nr, nc = agent_rc[0] + dr, agent_rc[1] + dc
        if action != 0 and (0 <= nr < n and 0 <= nc < n) and maze[nr, nc] == 0 and (nr, nc) != block_rc:
            agent_rc = (nr, nc)
        inside = _chebyshev(agent_rc, block_rc) <= contact_radius
        if inside and not was_inside:
            outside = [c for c in cells if _chebyshev(c, agent_rc) > contact_radius]
            if not outside:
                raise RuntimeError("no cell outside the contact radius")
            block_rc = outside[int(rng.integers(0, len(outside)))]
            contact[t] = 1
            inside = False
        was_inside = inside

    obs = _obs_from_positions(agent, block, n, noise_std, rng)
    return {
        "obs": obs,
        "agent": agent,
        "block": block,
        "action": actions,
        "contact": contact,
    }


def render_path(
    maze: np.ndarray,
    path: list[tuple[int, int]],
    block_rc: tuple[int, int],
    *,
    seed: int,
    noise_std: float,
) -> dict[str, np.ndarray]:
    """Noiseless-or-noisy frames of a scripted agent walk. The block stays put, off the path."""
    if block_rc in path:
        raise ValueError("scripted path runs through the parked block")
    n = int(maze.shape[0])
    length = len(path)
    agent = np.asarray(path, dtype=np.int64)
    block = np.tile(np.asarray(block_rc, dtype=np.int64), (length, 1))
    rng = np.random.default_rng(seed)
    obs = _obs_from_positions(agent, block, n, noise_std, rng)
    return {"obs": obs, "agent": agent, "block": block}


def normalize_rc(rc: np.ndarray, n: int) -> np.ndarray:
    """Map cell indices to [-1, 1]."""
    return rc.astype(np.float64) / float(n - 1) * 2.0 - 1.0


def chebyshev(agent: np.ndarray, block: np.ndarray) -> np.ndarray:
    return np.maximum(np.abs(agent[:, 0] - block[:, 0]), np.abs(agent[:, 1] - block[:, 1]))


def free_object_mask(agent: np.ndarray, block: np.ndarray, min_chebyshev: int) -> np.ndarray:
    """Frames where the block did not just move and the agent is not beside it.

    Matches the report's free-block readout: resting object, agent far from contact.
    """
    moved = np.zeros(agent.shape[0], dtype=bool)
    moved[1:] = np.any(block[1:] != block[:-1], axis=1)
    far = chebyshev(agent, block) >= min_chebyshev
    mask = (~moved) & far
    mask[0] = False
    return mask


def world_report(traj: dict[str, np.ndarray], min_chebyshev: int) -> dict[str, float]:
    agent = traj["agent"]
    block = traj["block"]
    mask = free_object_mask(agent, block, min_chebyshev)
    return {
        "n_frames": int(agent.shape[0]),
        "n_contacts": int(traj["contact"].sum()),
        "n_unique_block": int(np.unique(block, axis=0).shape[0]),
        "n_unique_agent": int(np.unique(agent, axis=0).shape[0]),
        "block_row_std": float(block[:, 0].astype(np.float64).std()),
        "block_col_std": float(block[:, 1].astype(np.float64).std()),
        "n_free_object": int(mask.sum()),
    }


def assert_maze_geometry(n: int = 7) -> None:
    maze = make_maze(n)
    index, dist = geodesic_table(maze)
    start = index[(1, 1)]
    other = index[(n - 2, 1)]
    manhattan = abs((n - 2) - 1) + abs(1 - 1)
    if dist[start, other] <= manhattan:
        raise AssertionError("geodesic should exceed Manhattan across the wall")
    route = scripted_route(maze)
    if len(route) < 8:
        raise AssertionError("scripted route is too short to score jaggedness")
    parked = parked_block(maze, route)
    if parked in route:
        raise AssertionError("parked block lies on the scripted route")
