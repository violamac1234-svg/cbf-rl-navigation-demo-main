# naive_navigation_env.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional, Dict, Any, Tuple, Union, List
from rsl_rl.env.vec_env import VecEnv  # Import VecEnv base class
import torch


class UnifiedNavigationEnv(VecEnv):
    """
    Custom vectorized environment for 2D robot navigation with single integrator dynamics.
    Uses PyTorch tensors for internal state representation.

    The agent controls a point robot by setting its velocity [vx, vy].
    The goal is to navigate to a target location while avoiding static circular obstacles.

    **Action Space:** Box(2,) - Velocity command [vx, vy] (handled as tensor)
    **Observation Space (Dict of Tensors):**
        - 'robot_pos': Tensor(num_envs, 2) - Robot's current [x, y] position.
        - 'obstacles': Tensor(num_envs, num_obstacles * 3) - Concatenated [x, y, radius] for each obstacle.
        - 'goal_pos': Tensor(num_envs, 2) - Goal's target [x, y] position.
        - 'last_velocity': Tensor(num_envs, 2) - Last commanded velocity [vx, vy].
    **Reward:** Combination of goal achievement bonus, collision penalty, and distance shaping (tensor).
    **Termination:** Episode ends if the robot reaches the goal or collides with an obstacle (tensor).
    **Truncation:** Can be handled by wrappers (e.g., TimeLimit).
    """

    # metadata and gym.Env inheritance removed

    def __init__(
        self,
        render_mode: Optional[str] = None,
        world_size: float = 10.0,
        num_obstacles: int = 3,
        robot_radius: float = 0.2,
        obstacle_radius: float = 0.5,
        goal_radius: float = 0.3,
        max_velocity: float = 1.0,
        dt: float = 0.1,
        max_episode_steps: Optional[int] = None,
        num_envs: int = 1,
        num_actions: int = 2,
        device: str = "cpu",  # Add device argument
        use_cbf_action_filtering: bool = True,
        use_cbf_reward_penalty: bool = True,
        noise_level: float = 0.0,
        cbf_alpha: float = 1.0,
        cbf_reward_weight: float = 100.0,
        cbf_sigma: float = 0.5,
    ):
        """
        Initializes the UnifiedNavigationEnv.

        Args:
            render_mode: The rendering mode ('human' or 'rgb_array').
            world_size: The size of the square world (from 0 to world_size).
            num_obstacles: The number of static obstacles.
            robot_radius: The radius of the point robot for collision checking.
            obstacle_radius: The radius of the obstacles. If float, all obstacles have this radius.
                             If list/array, must match num_obstacles.
            goal_radius: The radius of the goal area for determining success.
            max_velocity: The maximum absolute velocity allowed in x and y directions.
            dt: The time step duration for simulation.
            max_episode_steps: Maximum steps before truncation (informational, use wrapper).
            num_envs: Number of parallel environments.
            num_actions: Dimension of the action space.
            device: PyTorch device ('cpu' or 'cuda').
        """
        super().__init__()
        self.noise_level = noise_level
        self.cbf_alpha = float(cbf_alpha)
        self.cbf_reward_weight = float(cbf_reward_weight)
        self.cbf_sigma = float(cbf_sigma)
        self.world_size = float(world_size)
        self.num_obstacles = int(num_obstacles)
        self.robot_radius = float(robot_radius)
        self.goal_radius = float(goal_radius)
        self.max_velocity = float(max_velocity)
        self.dt = float(dt)
        self._max_episode_steps = max_episode_steps
        self.num_envs = num_envs
        self.device = device  # Store device
        self.num_actions = num_actions
        self.max_episode_length = (
            max_episode_steps if max_episode_steps is not None else 1000
        )
        self.use_cbf_action_filtering = use_cbf_action_filtering
        self.use_cbf_reward_penalty = use_cbf_reward_penalty

        # Handle obstacle radii - Store as tensor (num_envs, num_obstacles)
        if isinstance(obstacle_radius, (list, np.ndarray, torch.Tensor)):
            # Expecting shape (num_envs, num_obstacles) or (num_obstacles,) to broadcast
            obstacle_radii_np = (
                np.array(obstacle_radius, dtype=np.float32)
                if not isinstance(obstacle_radius, torch.Tensor)
                else obstacle_radius.cpu().numpy()
            )
            if obstacle_radii_np.ndim == 1:  # Shape (num_obstacles,)
                if len(obstacle_radii_np) != self.num_obstacles:
                    raise ValueError(
                        f"Length of 1D obstacle_radius ({len(obstacle_radii_np)}) "
                        f"must match num_obstacles ({self.num_obstacles})"
                    )
                # Broadcast to (num_envs, num_obstacles)
                self._obstacle_radii = (
                    torch.from_numpy(obstacle_radii_np)
                    .float()
                    .unsqueeze(0)
                    .expand(self.num_envs, -1)
                    .to(self.device)
                )
            elif obstacle_radii_np.ndim == 2:  # Shape (num_envs, num_obstacles)
                if obstacle_radii_np.shape != (self.num_envs, self.num_obstacles):
                    raise ValueError(
                        f"Shape of 2D obstacle_radius {obstacle_radii_np.shape} "
                        f"must match (num_envs, num_obstacles) ({self.num_envs}, {self.num_obstacles})"
                    )
                self._obstacle_radii = (
                    torch.from_numpy(obstacle_radii_np).float().to(self.device)
                )
            else:
                raise ValueError(
                    "obstacle_radius must be a float, 1D array/list/tensor, or 2D array/list/tensor"
                )

        else:  # Float case
            self._obstacle_radii = torch.full(
                (self.num_envs, self.num_obstacles),
                float(obstacle_radius),
                dtype=torch.float32,
                device=self.device,
            )

        # Max radius across all envs and obstacles
        self._max_obstacle_radius = (
            torch.max(self._obstacle_radii).item() if self.num_obstacles > 0 else 0.0
        )

        # --- Remove Gymnasium spaces ---
        # self.action_space = ...
        # self.observation_space = ...

        # --- Internal state variables (vectorized tensors) ---
        self._robot_pos: Optional[torch.Tensor] = None  # shape: (num_envs, 2)
        self._goal_pos: Optional[torch.Tensor] = None  # shape: (num_envs, 2)
        self._obstacle_positions: Optional[torch.Tensor] = (
            None  # shape: (num_envs, num_obstacles, 2)
        )
        self._last_velocity: torch.Tensor = torch.zeros(
            (self.num_envs, 2), dtype=torch.float32, device=self.device
        )
        self._elapsed_steps: Optional[torch.Tensor] = None  # shape: (num_envs,)
        self.episode_length_buf: torch.Tensor = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )

        # --- Rendering setup ---
        self.render_mode = render_mode
        self.figure: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None
        self.robot_patch: Optional[patches.Circle] = None
        self.goal_patch: Optional[patches.Circle] = None
        self.obstacle_patches: List[patches.Circle] = None
        # Store obstacle radii as numpy for rendering patch creation and reset logic
        self._obstacle_radii_np = (
            self._obstacle_radii.cpu().numpy()
        )  # Shape (num_envs, num_obstacles)
        self._episode_reward_components = {
            'reward_goal': torch.zeros(self.num_envs, device=self.device),
            'reward_obstacle_collision': torch.zeros(self.num_envs, device=self.device),
            'reward_wall_collision': torch.zeros(self.num_envs, device=self.device),
            'reward_progress': torch.zeros(self.num_envs, device=self.device),
            'reward_alive': torch.zeros(self.num_envs, device=self.device),
        }

        # assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.reset()  # Call reset to initialize state tensors

    # @property
    # def num_steps_per_env(self):
    #     return self.max_episode_length

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:  # Return tensor obs
        """
        Resets the environment to an initial state.

        Args:
            seed: The random seed for reproducibility (for NumPy placement).
            options: Additional options (e.g., specifying start/goal positions).

        Returns:
            A tuple containing the initial observation tensor and extras dictionary.
        """
        if seed is not None:
            np.random.seed(seed)  # Seed NumPy for placement
            # Consider seeding torch as well if torch.rand is used elsewhere
            # torch.manual_seed(seed)
            self.np_random = np.random
        else:
            self.np_random = np.random

        # Reset state tensors
        self._last_velocity = torch.zeros(
            (self.num_envs, 2), dtype=torch.float32, device=self.device
        )
        self._elapsed_steps = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        self.episode_length_buf = torch.zeros_like(
            self._elapsed_steps, dtype=torch.int32
        )  # Use zeros_like

        # Initialize state tensors
        self._robot_pos = torch.zeros(
            (self.num_envs, 2), dtype=torch.float32, device=self.device
        )
        self._goal_pos = torch.zeros(
            (self.num_envs, 2), dtype=torch.float32, device=self.device
        )
        self._obstacle_positions = torch.zeros(
            (self.num_envs, self.num_obstacles, 2),
            dtype=torch.float32,
            device=self.device,
        )

        # Use NumPy for placement logic, then convert to tensors
        robot_pos_np = np.zeros((self.num_envs, 2), dtype=np.float32)
        goal_pos_np = np.zeros((self.num_envs, 2), dtype=np.float32)
        obstacle_positions_np = np.zeros(
            (self.num_envs, self.num_obstacles, 2), dtype=np.float32
        )
        obstacle_radii_np = (
            self._obstacle_radii_np
        )  # Use the stored numpy version (num_envs, num_obstacles)

        for env_idx in range(self.num_envs):
            placement_attempts = 0
            max_placement_attempts = 100  # Increased attempts might be needed
            valid_placement = False
            min_obstacle_separation_buffer = (
                0.5  # Define minimum buffer between obstacles
            )
            min_robot_goal_distance = (
                self.world_size / 3.0
            )  # Minimum distance between robot and goal start
            # Get radii for the current environment
            current_env_obstacle_radii = obstacle_radii_np[
                env_idx
            ]  # Shape (num_obstacles,)
            current_env_max_obstacle_radius = (
                np.max(current_env_obstacle_radii) if self.num_obstacles > 0 else 0.0
            )

            while not valid_placement and placement_attempts < max_placement_attempts:
                placement_attempts += 1

                # 1. Place Obstacles (using NumPy)
                if options and options.get("obstacle_pos") is not None:
                    obs_pos = np.array(options["obstacle_pos"], dtype=np.float32)
                elif self.num_obstacles > 0:
                    obs_pos = self.np_random.uniform(
                        0
                        + current_env_max_obstacle_radius,  # Use env-specific max radius
                        self.world_size
                        - current_env_max_obstacle_radius,  # Use env-specific max radius
                        size=(self.num_obstacles, 2),
                    ).astype(np.float32)
                else:
                    obs_pos = np.empty((0, 2), dtype=np.float32)

                # 2. Place Goal (using NumPy)
                if options and options.get("goal_pos") is not None:
                    goal_pos = np.array(options["goal_pos"], dtype=np.float32)
                else:
                    # Use a buffer that respects both goal_radius and robot_radius to avoid h < 0 at walls
                    goal_wall_buffer = max(self.goal_radius, self.robot_radius)
                    goal_pos = self.np_random.uniform(
                        0 + goal_wall_buffer,
                        self.world_size - goal_wall_buffer,
                        size=(2,),
                    ).astype(np.float32)

                # 3. Place Robot (using NumPy)
                if options and options.get("robot_pos") is not None:
                    robot_pos = np.array(options["robot_pos"], dtype=np.float32)
                else:
                    # Sample strictly inside the walls to ensure h_wall > 1.0
                    robot_wall_buffer = self.robot_radius + 1.0 + 1e-4
                    robot_pos = self.np_random.uniform(
                        0 + robot_wall_buffer,
                        self.world_size - robot_wall_buffer,
                        size=(2,),
                    ).astype(np.float32)

                # --- Check for Initial Collisions/Overlaps (using NumPy) ---
                # Ensure robot h > 1.0 wrt walls (strict)
                robot_hsafe = True
                wall_buffer = self.robot_radius + 1.0
                if (
                    (robot_pos[0] <= wall_buffer)
                    or (robot_pos[0] >= self.world_size - wall_buffer)
                    or (robot_pos[1] <= wall_buffer)
                    or (robot_pos[1] >= self.world_size - wall_buffer)
                ):
                    robot_hsafe = False
                if not robot_hsafe:
                    continue

                # Check Robot vs Obstacles (strict >, i.e., reject if <= r_robot + r_obst + 1.0)
                robot_clear = True
                for i, o_pos in enumerate(obs_pos):
                    if (
                        np.linalg.norm(robot_pos - o_pos)
                        <= self.robot_radius + current_env_obstacle_radii[i] + 1.0
                    ):
                        robot_clear = False
                        break
                if not robot_clear:
                    continue

                # Check Robot vs Goal (minimum distance)
                # goal_clear_robot = np.linalg.norm(robot_pos - goal_pos) > self.robot_radius + self.goal_radius
                robot_goal_dist = np.linalg.norm(robot_pos - goal_pos)
                if robot_goal_dist < min_robot_goal_distance:
                    continue  # Retry placement if robot and goal are too close

                # New: ensure goal is in an area where CBF h >= 0 (safe w.r.t. obstacles and walls)
                goal_hsafe = True
                # Walls check for goal center using robot_radius (h_wall >= 0)
                if (
                    (goal_pos[0] < self.robot_radius)
                    or (goal_pos[0] > self.world_size - self.robot_radius)
                    or (goal_pos[1] < self.robot_radius)
                    or (goal_pos[1] > self.world_size - self.robot_radius)
                ):
                    goal_hsafe = False
                # Obstacles check for goal center using robot_radius + obstacle_radius (h_obs >= 0)
                if goal_hsafe and self.num_obstacles > 0:
                    for i, o_pos in enumerate(obs_pos):
                        if (
                            np.linalg.norm(goal_pos - o_pos)
                            < (self.robot_radius + current_env_obstacle_radii[i])
                        ):
                            goal_hsafe = False
                            break
                if not goal_hsafe:
                    continue

                # Check Goal vs Obstacles (existing, using goal_radius) - keep for extra clearance
                goal_clear_obstacles = True
                for i, o_pos in enumerate(obs_pos):
                    if (
                        np.linalg.norm(goal_pos - o_pos)
                        < self.goal_radius + current_env_obstacle_radii[i]
                    ):
                        goal_clear_obstacles = False
                        break
                if not goal_clear_obstacles:
                    continue

                # Check Obstacles vs Obstacles (minimum separation)
                obstacles_clear = True
                if self.num_obstacles > 1:
                    for i in range(self.num_obstacles):
                        for j in range(i + 1, self.num_obstacles):
                            dist_sq = np.sum((obs_pos[i] - obs_pos[j]) ** 2)
                            # Use env-specific radii + buffer for minimum separation check
                            min_dist_sq = (
                                current_env_obstacle_radii[i]
                                + current_env_obstacle_radii[j]
                                + min_obstacle_separation_buffer
                            ) ** 2
                            if dist_sq < min_dist_sq:
                                obstacles_clear = False
                                break
                        if not obstacles_clear:
                            break
                    if not obstacles_clear:
                        continue  # Try placing obstacles again if too close

                # Check if at least one obstacle is near the robot-goal line
                obstacle_blocks_path = False
                if self.num_obstacles > 0:
                    dist_robot_goal = np.linalg.norm(robot_pos - goal_pos)
                    for i, o_pos in enumerate(obs_pos):
                        dist_robot_obs = np.linalg.norm(robot_pos - o_pos)
                        dist_goal_obs = np.linalg.norm(goal_pos - o_pos)
                        # Check if obstacle center is roughly between robot and goal
                        # Use env-specific radius
                        if (
                            abs(dist_robot_obs + dist_goal_obs - dist_robot_goal)
                            < current_env_obstacle_radii[i] * 2
                        ):
                            obstacle_blocks_path = True
                            break  # Found one obstacle on the path
                    if not obstacle_blocks_path:
                        continue  # Retry placement if no obstacle blocks the path
                else:
                    obstacle_blocks_path = (
                        True  # No obstacles, so condition is trivially met
                    )

                # If all checks pass (including the new h-safe goal check)
                valid_placement = True

            if not valid_placement:
                print(
                    f"Warning: Failed to find valid initial placement for env {env_idx} after {max_placement_attempts} attempts (incl. path block & min dist checks)."
                )
                # Handle failure? Maybe place at default corners? For now, just warn.

            robot_pos_np[env_idx] = robot_pos
            goal_pos_np[env_idx] = goal_pos
            if self.num_obstacles > 0:
                obstacle_positions_np[env_idx] = obs_pos

        # Convert placed positions to tensors and assign to state variables
        self._robot_pos = torch.from_numpy(robot_pos_np).to(self.device)
        self._goal_pos = torch.from_numpy(goal_pos_np).to(self.device)
        self._obstacle_positions = torch.from_numpy(obstacle_positions_np).to(
            self.device
        )

        # Get observations and info (now using tensor methods)
        obs, extras = self.get_observations()

        if self.render_mode == "human":
            self._render_frame()

        return obs, extras  # Return tensor obs and dict extras

    def _get_obs(self) -> Dict[str, torch.Tensor]:
        """Constructs the observation dictionary with tensors."""
        # Defensive: check state initialization
        if (
            self._robot_pos is None
            or self._goal_pos is None
            or self._last_velocity is None
            or self._obstacle_positions is None
        ):
            raise RuntimeError(
                "Environment state is uninitialized. Call reset() before using the environment."
            )

        obs_dict = {
            "robot_pos": self._robot_pos.clone(),
            "goal_pos": self._goal_pos.clone(),
            "last_velocity": self._last_velocity.clone(),
        }
        if self.num_obstacles > 0:
            # Reshape obstacle positions and radii for concatenation
            # Obstacle pos: (num_envs, num_obstacles, 2)
            # Radii: (num_envs, num_obstacles) -> (num_envs, num_obstacles, 1)
            obstacle_radii_expanded = self._obstacle_radii.unsqueeze(2)  # Add last dim
            # Concatenate pos and radii: (num_envs, num_obstacles, 3)
            # get the closest obstacle position relative to the robot position
            relative_obstacle_positions = self._obstacle_positions - self._robot_pos.unsqueeze(1)
            obstacle_distances = torch.linalg.norm(relative_obstacle_positions, dim=2)
            idx_min_distances = torch.argmin(obstacle_distances, dim=1)
            closest_obstacle_positions = self._obstacle_positions[torch.arange(self.num_envs), idx_min_distances]
            closest_obstacle_radii = obstacle_radii_expanded[torch.arange(self.num_envs), idx_min_distances]
            # print("closest_obstacle_positions:", closest_obstacle_positions.unsqueeze(1).shape)
            # print("self._obstacle_positions:", self._obstacle_positions.shape)
            # print("obstacle_radii_expanded:", obstacle_radii_expanded.shape)
            obstacle_info_tensor = torch.cat(
                (closest_obstacle_positions.unsqueeze(1), closest_obstacle_radii.unsqueeze(1)), dim=2
            )
            # obstacle_info_tensor = torch.cat(
            #     (self._obstacle_positions, obstacle_radii_expanded), dim=2
            # )
            # Flatten: (num_envs, num_obstacles * 3)
            obs_dict["obstacles"] = obstacle_info_tensor.view(self.num_envs, -1)
        else:
            obs_dict["obstacles"] = torch.empty(
                (self.num_envs, 0), dtype=torch.float32, device=self.device
            )
        return obs_dict

    def _get_info(self) -> Dict[str, Any]:
        """Provides auxiliary information about the environment state (tensors where applicable)."""
        # Defensive: check state initialization
        if (
            self._robot_pos is None
            or self._goal_pos is None
            or self._obstacle_positions is None
            or self._elapsed_steps is None
        ):
            raise RuntimeError(
                "Environment state is uninitialized. Call reset() before using the environment."
            )

        dist_to_goal = torch.linalg.norm(self._robot_pos - self._goal_pos, dim=1)

        return {
            "distance_to_goal": dist_to_goal,  # Tensor
            "robot_position": self._robot_pos.clone(),  # Tensor
            "goal_position": self._goal_pos.clone(),  # Tensor
            "obstacle_positions": self._obstacle_positions.clone(),  # Tensor
            "elapsed_steps": self._elapsed_steps.clone(),  # Tensor
        }

    def get_observations(self):
        """
        Returns the current observations and extras in the format expected by rsl_rl VecEnv:
        - obs: torch.Tensor of shape (num_envs, obs_dim) on self.device
        - extras: dict with key 'observations' containing the observation dict (tensors on self.device)
                  and other info (tensors on self.device)
        """
        obs_dict = self._get_obs()  # Returns dict of tensors
        # Flatten and concatenate all observation components for each env
        obs_list = []
        # Ensure consistent order for concatenation
        for k in sorted(obs_dict.keys()):
            # Ensure tensors are 2D (num_envs, feature_dim) before concatenating
            tensor = obs_dict[k]
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(1)
            elif tensor.ndim > 2:  # Should not happen with current obs space
                tensor = tensor.view(self.num_envs, -1)
            obs_list.append(tensor)

        obs = torch.cat(obs_list, dim=1)  # shape (num_envs, obs_dim)

        # Get info dict (already contains tensors where applicable)
        info_dict = self._get_info()

        extras = {"observations": obs_dict}  # obs_dict already contains tensors
        extras.update(info_dict)  # info_dict already contains tensors
        return obs, extras

    def h_function(
        self,
        robot_pos: torch.Tensor,
        obstacle_pos: torch.Tensor,
        obstacle_radius: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算论文式 (24) 的统一障碍函数 h(q)。

        每个圆形障碍物和四面墙各产生一个“净安全距离”，取其中最小值
        作为当前真正起作用的约束。这样后续闭式过滤器只需求解一个约束。
        h is the minimum across all active constraints (obstacles and 4 walls).
        Positive => safe, 0 => on boundary, negative => unsafe.
        """
        device = robot_pos.device
        num_envs = robot_pos.shape[0]

        # Obstacles part (handle num_obstacles == 0 safely)
        if obstacle_pos is not None and obstacle_pos.shape[1] > 0:
            # distances: (num_envs, num_obstacles)
            distances = torch.linalg.norm(robot_pos.unsqueeze(1) - obstacle_pos, dim=2)
            # h for each obstacle: d - (r_robot + r_obstacle)
            h_obs_all = distances - (self.robot_radius + obstacle_radius)
            # min across obstacles
            h_obs, _ = torch.min(h_obs_all, dim=1)  # (num_envs,)
        else:
            # No obstacles -> do not constrain via obstacles
            h_obs = torch.full((num_envs,), float("inf"), device=device, dtype=robot_pos.dtype)

        # Walls part (distance from robot center to walls minus robot radius)
        x = robot_pos[:, 0]
        y = robot_pos[:, 1]
        h_left = x - self.robot_radius
        h_right = (self.world_size - x) - self.robot_radius
        h_bottom = y - self.robot_radius
        h_top = (self.world_size - y) - self.robot_radius
        h_wall = torch.minimum(torch.minimum(h_left, h_right), torch.minimum(h_bottom, h_top))

        # Combine: active constraint is the most critical one (minimum h)
        h = torch.minimum(h_obs, h_wall)  # (num_envs,)
        return h

    def gradient_h_function(
        self, robot_pos: torch.Tensor, obstacle_pos: torch.Tensor
    ) -> torch.Tensor:
        """
        计算论文式 (25) 中当前活动约束的梯度 ∇h(q)。
        - For obstacles: unit vector from closest obstacle center to the robot.
        - For walls: outward unit normal of the closest wall.
        """
        device = robot_pos.device
        dtype = robot_pos.dtype
        num_envs = robot_pos.shape[0]

        # Compute obstacle-side h and gradient candidates (if any obstacles)
        has_obstacles = obstacle_pos is not None and obstacle_pos.shape[1] > 0
        if has_obstacles:
            # distances: (num_envs, num_obstacles)
            distances = torch.linalg.norm(robot_pos.unsqueeze(1) - obstacle_pos, dim=2)
            # h for each obstacle uses per-env obstacle radii
            # Use self._obstacle_radii since signature does not include radii
            h_obs_all = distances - (self.robot_radius + self._obstacle_radii)
            # pick the obstacle with minimum h
            min_h_obs, min_idx = torch.min(h_obs_all, dim=1)  # (num_envs,), (num_envs,)
            closest_obs = obstacle_pos[torch.arange(num_envs, device=device), min_idx]  # (num_envs,2)
            # 障碍物中心处距离为零会导致除零；clamp_min 仅用于数值保护。
            denom = distances[torch.arange(num_envs, device=device), min_idx].clamp_min(1e-8).unsqueeze(1)
            grad_obs = (robot_pos - closest_obs) / denom  # (num_envs,2)
        else:
            # No obstacles: set min_h_obs to +inf and grad_obs dummy
            min_h_obs = torch.full((num_envs,), float("inf"), device=device, dtype=dtype)
            grad_obs = torch.zeros((num_envs, 2), device=device, dtype=dtype)

        # Compute wall h values and gradients
        x = robot_pos[:, 0]
        y = robot_pos[:, 1]
        h_vals = torch.stack(
            [
                x - self.robot_radius,                           # left wall (grad [1,0])
                (self.world_size - x) - self.robot_radius,       # right wall (grad [-1,0])
                y - self.robot_radius,                           # bottom wall (grad [0,1])
                (self.world_size - y) - self.robot_radius,       # top wall (grad [0,-1])
            ],
            dim=1,
        )  # (num_envs, 4)
        min_h_wall, wall_idx = torch.min(h_vals, dim=1)  # (num_envs,), (num_envs,)

        wall_grads_lut = torch.tensor(
            [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
            device=device,
            dtype=dtype,
        )  # order matches h_vals stacking
        grad_wall = wall_grads_lut[wall_idx]  # (num_envs,2)

        # Select active gradient based on which constraint is more critical
        use_obstacle = min_h_obs <= min_h_wall
        grad_h = torch.where(use_obstacle.unsqueeze(1), grad_obs, grad_wall)
        return grad_h

    def filter_velocity(
        self,
        robot_pos: torch.Tensor,
        velocity: torch.Tensor,
        obstacle_pos: torch.Tensor,
        obstacle_radius: torch.Tensor,
    ) -> torch.Tensor:
        """
        论文式 (20) 的闭式 CBF 安全过滤器（对应单约束 CBF-QP）。

        输入 velocity 是策略提出并经过速度裁剪的名义动作 v_policy；输出
        filtered_velocity 是满足 ∇h(q)^T v + alpha*h(q) >= 0 的 v_safe。
        psi 是过滤前的 CBF 条件值：psi < 0 表示必须启动过滤器。
        """
        # 1) 找到最危险的障碍物/墙壁，并取得该约束的 h 与梯度。
        h = self.h_function(robot_pos, obstacle_pos, obstacle_radius)            # (num_envs,)
        grad_h = self.gradient_h_function(robot_pos, obstacle_pos)               # (num_envs,2)

        # 2) 计算 psi = ∇h^T v_policy + alpha*h。psi >= 0 时动作本来就安全。
        u_des = velocity
        Lgh_u_des = torch.sum(grad_h * u_des, dim=1)                             # (num_envs,)
        psi = Lgh_u_des + self.cbf_alpha * h                                     # (num_envs,)

        # 3) 仅对 psi < 0 的并行环境应用最小二范数修正：
        #    v_safe = v_policy - psi * ∇h / ||∇h||^2。
        filtered_velocity = velocity.clone()
        filtered_ids = torch.where(psi < 0)[0]
        if filtered_ids.numel() > 0:
            # 理论上活动约束梯度非零；下限用于防止浮点异常。
            denom = torch.sum(grad_h[filtered_ids] ** 2, dim=1).clamp_min(1e-12) # (k,)
            correction = (-psi[filtered_ids] / denom).unsqueeze(1) * grad_h[filtered_ids]
            filtered_velocity[filtered_ids] += correction

        return filtered_velocity, psi

    def step(self, action: torch.Tensor):
        """
        Executes one time step in the environment using PyTorch tensors.

        Args:
            action: Tensor of shape (num_envs, num_actions) on self.device.

        Returns:
            obs: Observation tensor (num_envs, obs_dim).
            reward: Reward tensor (num_envs,).
            done: Done tensor (num_envs,).
            extras: Dictionary containing detailed observation, info, log, and episode data (tensors).
        """
        # Action is expected to be a tensor on self.device
        if action.device != self.device:
            action = action.to(self.device)

        # ===== 动作数据流 =====
        # policy action -> 速度范围裁剪 -> 计算 CBF 安全动作 -> 选择实际执行动作。
        # 注意：即使本方法不执行过滤动作，也必须计算 v_safe 和 psi；安全奖励需要它们。
        clipped_action = torch.clamp(action, -self.max_velocity, self.max_velocity)
        filtered_action, psi = self.filter_velocity(
            self._robot_pos,
            clipped_action,
            self._obstacle_positions,
            self._obstacle_radii,
        )

        # Filter Only/Dual 在训练时执行 v_safe；Nominal/Reward Only 执行 v_policy。
        if self.use_cbf_action_filtering:
            self._last_velocity = filtered_action
        else:
            self._last_velocity = clipped_action
        prev_dist_to_goal = torch.linalg.norm(self._robot_pos - self._goal_pos, dim=1)

        # Paper DR: independent standard-normal velocity disturbance, scaled by
        # a fraction (0.2 in the reported experiment) of maximum velocity.
        rand_velocity = torch.randn_like(self._last_velocity) * self.max_velocity * self.noise_level
        new_pos = self._robot_pos + (self._last_velocity + rand_velocity) * self.dt
        # Clamp position to stay within world boundaries (center of robot)
        # Note: Clamping here prevents the *center* from going out, but collision check below handles radius overlap
        self._robot_pos = torch.clamp(
            new_pos,
            0.0,  # Clamp center to 0
            self.world_size,  # Clamp center to world_size
        )
        self._elapsed_steps += 1
        self.episode_length_buf = self._elapsed_steps.clone().int()  # Update buffer

        # --- Vectorized termination/collision/goal logic (using tensor operations) ---
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        collided_obstacle = torch.zeros_like(terminated)  # Renamed from collided
        goal_reached = torch.zeros_like(terminated)
        wall_collision = torch.zeros_like(terminated)  # New tensor for wall collision

        # Obstacle Collision check
        if self.num_obstacles > 0:
            # Expand dims for broadcasting:
            # robot_pos: (num_envs, 1, 2)
            # obstacle_pos: (num_envs, num_obstacles, 2)
            # obstacle_radii: (num_envs, num_obstacles) - No expansion needed
            robot_pos_exp = self._robot_pos.unsqueeze(1)
            obstacle_pos_exp = self._obstacle_positions
            obstacle_radii_exp = (
                self._obstacle_radii
            )  # Already (num_envs, num_obstacles)

            # Calculate distances: (num_envs, num_obstacles)
            distances_sq = torch.sum((robot_pos_exp - obstacle_pos_exp) ** 2, dim=2)
            # Collision thresholds: (num_envs, num_obstacles)
            collision_thresholds = (
                self.robot_radius + obstacle_radii_exp
            ) ** 2  # Use per-env radii
            # Check collision: (num_envs, num_obstacles)
            collisions_per_obstacle = distances_sq < collision_thresholds
            # Any collision per env: (num_envs,)
            collided_obstacle = torch.any(collisions_per_obstacle, dim=1)
            terminated = (
                terminated | collided_obstacle
            )  # Update terminated based on obstacle collision

        # Wall Collision check
        # Check if robot center is too close to any boundary
        wall_collision = (
            (self._robot_pos[:, 0] < self.robot_radius)
            | (self._robot_pos[:, 0] > self.world_size - self.robot_radius)
            | (self._robot_pos[:, 1] < self.robot_radius)
            | (self._robot_pos[:, 1] > self.world_size - self.robot_radius)
        )
        # Only count wall collision if not already terminated by obstacle
        wall_collision = wall_collision & (~terminated)
        terminated = (
            terminated | wall_collision
        )  # Update terminated based on wall collision

        # Goal check
        current_dist_to_goal = torch.linalg.norm(
            self._robot_pos - self._goal_pos, dim=1
        )
        goal_reached = current_dist_to_goal < (self.robot_radius + self.goal_radius)
        # Only count goal reached if not already terminated by collision (obstacle or wall)
        goal_reached = goal_reached & (~terminated)
        terminated = terminated | goal_reached  # Update terminated based on goal

        # --- Vectorized reward calculation (using tensor operations) ---
        # Normalize rewards: Goal=1.0, Collisions=-1.0
        reward_goal = goal_reached.float() * 1.0  # Normalized goal reward
        reward_obstacle_collision = (
            collided_obstacle.float() * -1.0
        )  # Normalized obstacle collision penalty
        reward_wall_collision = (
            wall_collision.float() * -1.0
        )  # Normalized wall collision penalty
        

        # Progress reward: Normalize by max possible progress per step
        active_mask = ~terminated  # Mask for steps not ended by collision or goal
        max_progress_per_step = (
            self.max_velocity * self.dt + 1e-8
        )  # Add epsilon for stability
        reward_progress = 20 * (
            (prev_dist_to_goal - current_dist_to_goal) / max_progress_per_step
        ) * active_mask.float()

        # Table II calls this an alive reward and assigns +0.01.
        reward_alive = 0.01 * active_mask.float()

        # ===== 论文式 (22)-(23)：安全奖励 r_cbf =====
        # 两项都使用原始策略动作计算，而环境可执行 filtered_action。
        # 第一项惩罚 CBF 条件违反；第二项惩罚安全过滤器的修正幅度。
        if self.use_cbf_reward_penalty:
            correction_sq = torch.sum(torch.square(clipped_action - filtered_action), dim=1)
            # 100 * (exp(-||v_policy-v_safe||^2 / sigma^2) - 1)，值域 [-100, 0]。
            reward_clipped_action = self.cbf_reward_weight * (
                torch.exp(-correction_sq / (self.cbf_sigma ** 2)) - 1.0
            ) * active_mask.float()
            # 100 * min(∇h^T v_policy + alpha*h, 0)。安全动作对应零惩罚。
            reward_psi = self.cbf_reward_weight * torch.clamp(psi, max=0.0) * active_mask.float()
        else:
            reward_clipped_action = torch.zeros_like(reward_goal)
            reward_psi = torch.zeros_like(reward_goal)

        # 总奖励 = 任务奖励 + 正则/存活奖励 + CBF 安全奖励。
        reward = (
            reward_goal
            + reward_obstacle_collision
            + reward_wall_collision
            + reward_progress
            + reward_alive
            + reward_clipped_action
            + reward_psi
        )

        reward_log = (
            reward_goal
            + reward_obstacle_collision
            + reward_wall_collision
            + reward_progress
            + reward_alive
            + reward_clipped_action
            # + reward_psi
        )

        # --- Truncation ---
        truncated = torch.zeros_like(terminated)
        if self._max_episode_steps is not None:
            truncated = self._elapsed_steps >= self._max_episode_steps
        # Reset episode length buffer for terminated/truncated envs
        self.episode_length_buf[terminated | truncated] = 0
        # Reset elapsed steps for terminated/truncated envs
        self._elapsed_steps[terminated | truncated] = 0
        timeout_mask = truncated & (~terminated)
        reward_timeout = timeout_mask.float() * -10.0
        reward = reward + reward_timeout  # apply timeout penalty

        # --- Prepare outputs ---
        obs, extras = self.get_observations()  # Get tensor obs and extras dict
        done = terminated | truncated  # Combine termination and truncation

        # log 保存逐步指标；test.py 用这些量复现成功率、违反次数和修正量统计。
        extras["log"] = {
            "reward_goal": reward_goal,
            "reward_obstacle_collision": reward_obstacle_collision,
            "reward_wall_collision": reward_wall_collision,  # Add wall collision reward log
            "reward_progress": reward_progress,
            "reward_alive": reward_alive,
            "reward_psi": reward_psi,
            "reward_clipped_action": reward_clipped_action,
            "reward_timeout": reward_timeout,
            "success": goal_reached.float(),  # Use float for logging
            "collided_obstacle": collided_obstacle.float(),  # Renamed log key
            "collided_wall": wall_collision.float(),  # Add wall collision status log
            "clipped_action": clipped_action,
            "filtered_action": filtered_action,
            "psi": psi,
            "cbf_violated": (psi < 0).float(),
            "filter_activated": (psi < 0).float(),
            "action_correction_norm": torch.linalg.norm(filtered_action - clipped_action, dim=1),
            "min_safety_margin": self.h_function(self._robot_pos, self._obstacle_positions, self._obstacle_radii),
        }
        # Add episode info for runner logging (as tensors)
        extras["episode"] = {
            "reward": reward,  # Total reward per step
            "reward_log": reward_log,
            "length": self.episode_length_buf,  # Current episode length (already updated)
            "reward_goal": reward_goal,
            "reward_obstacle_collision": reward_obstacle_collision,
            "reward_wall_collision": reward_wall_collision,  # Add wall collision reward log
            "reward_progress": reward_progress,
            "reward_alive": reward_alive,
            "reward_clipped_action": reward_clipped_action,
            "reward_psi": reward_psi,
            "reward_timeout": reward_timeout,
            "success": goal_reached.float(),  # Use float for logging
            "collided_obstacle": collided_obstacle.float(),  # Renamed log key
            "collided_wall": wall_collision.float(),  # Add wall collision status log
            "clipped_action": clipped_action,  # Add clipped action log
            "filtered_action": filtered_action,
            "psi": psi,
            "cbf_violated": (psi < 0).float(),
            "filter_activated": (psi < 0).float(),
            "action_correction_norm": torch.linalg.norm(filtered_action - clipped_action, dim=1),
            
        }

        # Reset environments that are done
        # Note: rsl_rl typically handles resets externally based on 'done' flags.
        # If internal reset is needed, it would go here, potentially using indices from `done`.
        reset_indices = torch.where(done)[0]
        if len(reset_indices) > 0:
            self._reset_envs(reset_indices) # Need a helper function for partial resets

        # Return observations matching the post-reset internal state. Terminal
        # statistics remain available in extras assembled above.
        obs, post_reset_extras = self.get_observations()
        extras["observations"] = post_reset_extras["observations"]
        
        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, done, extras

    # --- Rendering methods ---
    def render(self) -> Optional[np.ndarray]:
        """
        Renders the environment. Converts tensors to NumPy for plotting.
        """
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()
            return None

    def _render_frame(self) -> Optional[np.ndarray]:
        """Internal rendering function using Matplotlib."""
        if self.render_mode is None:
            print(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization."
            )
            return None

        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
        except ImportError:
            print("matplotlib is not installed, run `pip install matplotlib`")
            return None

        if self.figure is None:  # Initialize plot on first call
            if self.render_mode == "human":
                plt.ion()  # Enable interactive mode for human rendering
                self.figure, self.ax = plt.subplots(figsize=(6, 6))
            elif self.render_mode == "rgb_array":
                # No interactive mode needed for array rendering
                self.figure, self.ax = plt.subplots(figsize=(6, 6))

            self.ax.set_xlim(0, self.world_size)
            self.ax.set_ylim(0, self.world_size)
            self.ax.set_aspect("equal", adjustable="box")
            self.ax.set_title("Custom Navigation Environment")
            self.ax.set_xlabel("X Position")
            self.ax.set_ylabel("Y Position")

            # Create patches (visual elements) only once
            self.robot_patch = patches.Circle(
                (0, 0), self.robot_radius, fc="blue", alpha=0.8, label="Robot"
            )
            self.goal_patch = patches.Circle(
                (0, 0), self.goal_radius, fc="green", alpha=0.8, label="Goal"
            )
            # Create obstacle patches using radii from the *first* environment for visualization
            first_env_radii_np = self._obstacle_radii_np[0]  # Shape (num_obstacles,)
            self.obstacle_patches = [
                patches.Circle((0, 0), radius, fc="red", alpha=0.6)
                for radius in first_env_radii_np  # Use numpy version for env 0
            ]

            self.ax.add_patch(self.robot_patch)
            self.ax.add_patch(self.goal_patch)
            for patch in self.obstacle_patches:
                self.ax.add_patch(patch)
            self.ax.legend(loc="upper right")

        # --- Update patch positions based on current state ---
        # Ensure state variables are initialized (should be after reset)
        if (
            self._robot_pos is None
            or self._goal_pos is None
            or self._obstacle_positions is None
        ):
            print("Warning: Attempting to render before reset() or with invalid state.")
            # Handle gracefully, e.g., return None or render default positions
            if self.render_mode == "rgb_array":
                return np.zeros((100, 100, 3), dtype=np.uint8)  # Placeholder
            else:
                return None

        # Convert tensors to NumPy for rendering (only for the first env if num_envs > 1)
        robot_pos_np = self._robot_pos[0].cpu().numpy()
        goal_pos_np = self._goal_pos[0].cpu().numpy()
        obstacle_positions_np = (
            self._obstacle_positions[0].cpu().numpy()
        )  # Shape (num_obstacles, 2)
        # Radii are already set in the patches during initialization based on env 0

        self.robot_patch.set_center(robot_pos_np)
        self.goal_patch.set_center(goal_pos_np)
        for i, patch in enumerate(self.obstacle_patches):
            if i < self.num_obstacles:
                patch.set_center(obstacle_positions_np[i])
            else:
                # If there are fewer obstacles than patches, set remaining patches to a default position
                patch.set_center((0, 0))
        # No need to update obstacle patch positions, they are set during init

        # --- Draw and return/display (unchanged) ---
        if self.render_mode == "human":
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()
            # plt.pause(0.001) # Small pause helps update plot in some backends
            return None
        elif self.render_mode == "rgb_array":
            self.figure.canvas.draw()
            image = np.frombuffer(self.figure.canvas.tostring_rgb(), dtype="uint8")
            width, height = self.figure.canvas.get_width_height()
            image = image.reshape(height, width, 3)
            # Note: Closing the figure here would prevent updates in subsequent calls.
            # Keep the figure object alive unless explicitly closing the env.
            return image
        else:
            return None  # Should not happen if render_mode is validated
    
    def _reset_envs(self, reset_indices: torch.Tensor):
        """
        Resets only the environments specified by reset_indices.

        Args:
            reset_indices: 1D torch.Tensor of indices to reset (on CPU or self.device).
        """
        if not isinstance(reset_indices, torch.Tensor):
            reset_indices = torch.tensor(reset_indices, dtype=torch.long)
        reset_indices = reset_indices.cpu().numpy()  # For numpy indexing

        # Use NumPy for placement logic, then convert to tensors
        robot_pos_np = self._robot_pos.cpu().numpy()
        goal_pos_np = self._goal_pos.cpu().numpy()
        obstacle_positions_np = self._obstacle_positions.cpu().numpy()
        obstacle_radii_np = self._obstacle_radii.cpu().numpy()

        for env_idx in reset_indices:
            placement_attempts = 0
            max_placement_attempts = 100
            valid_placement = False
            min_obstacle_separation_buffer = 0.5
            min_robot_goal_distance = self.world_size / 3.0
            current_env_obstacle_radii = obstacle_radii_np[env_idx]
            current_env_max_obstacle_radius = (
                np.max(current_env_obstacle_radii) if self.num_obstacles > 0 else 0.0
            )

            while not valid_placement and placement_attempts < max_placement_attempts:
                placement_attempts += 1

                # 1. Place Obstacles (using NumPy)
                if self.num_obstacles > 0:
                    obs_pos = np.random.uniform(
                        0 + current_env_max_obstacle_radius,
                        self.world_size - current_env_max_obstacle_radius,
                        size=(self.num_obstacles, 2),
                    ).astype(np.float32)
                else:
                    obs_pos = np.empty((0, 2), dtype=np.float32)

                # 2. Place Goal (using NumPy) with wall buffer
                goal_wall_buffer = max(self.goal_radius, self.robot_radius)
                goal_pos = np.random.uniform(
                    0 + goal_wall_buffer,
                    self.world_size - goal_wall_buffer,
                    size=(2,),
                ).astype(np.float32)

                # 3. Place Robot (using NumPy) ensuring h_wall > 1.0
                robot_wall_buffer = self.robot_radius + 1.0 + 1e-4
                robot_pos = np.random.uniform(
                    0 + robot_wall_buffer,
                    self.world_size - robot_wall_buffer,
                    size=(2,),
                ).astype(np.float32)

                # Ensure robot h > 1.0 wrt walls (strict)
                wall_buffer = self.robot_radius + 1.0
                if (
                    (robot_pos[0] <= wall_buffer)
                    or (robot_pos[0] >= self.world_size - wall_buffer)
                    or (robot_pos[1] <= wall_buffer)
                    or (robot_pos[1] >= self.world_size - wall_buffer)
                ):
                    continue

                # Robot vs Obstacles (strict > r_robot + r_obst + 1.0)
                robot_clear = True
                for i, o_pos in enumerate(obs_pos):
                    if (
                        np.linalg.norm(robot_pos - o_pos)
                        <= self.robot_radius + current_env_obstacle_radii[i] + 1.0
                    ):
                        robot_clear = False
                        break
                if not robot_clear:
                    continue

                robot_goal_dist = np.linalg.norm(robot_pos - goal_pos)
                if robot_goal_dist < min_robot_goal_distance:
                    continue

                # New: ensure goal is in an area where CBF h >= 0 (safe w.r.t. obstacles and walls)
                goal_hsafe = True
                if (
                    (goal_pos[0] < self.robot_radius)
                    or (goal_pos[0] > self.world_size - self.robot_radius)
                    or (goal_pos[1] < self.robot_radius)
                    or (goal_pos[1] > self.world_size - self.robot_radius)
                ):
                    goal_hsafe = False
                if goal_hsafe and self.num_obstacles > 0:
                    for i, o_pos in enumerate(obs_pos):
                        if (
                            np.linalg.norm(goal_pos - o_pos)
                            < (self.robot_radius + current_env_obstacle_radii[i])
                        ):
                            goal_hsafe = False
                            break
                if not goal_hsafe:
                    continue

                goal_clear_obstacles = True
                for i, o_pos in enumerate(obs_pos):
                    if (
                        np.linalg.norm(goal_pos - o_pos)
                        < self.goal_radius + current_env_obstacle_radii[i]
                    ):
                        goal_clear_obstacles = False
                        break
                if not goal_clear_obstacles:
                    continue

                obstacles_clear = True
                if self.num_obstacles > 1:
                    for i in range(self.num_obstacles):
                        for j in range(i + 1, self.num_obstacles):
                            dist_sq = np.sum((obs_pos[i] - obs_pos[j]) ** 2)
                            min_dist_sq = (
                                current_env_obstacle_radii[i]
                                + current_env_obstacle_radii[j]
                                + min_obstacle_separation_buffer
                            ) ** 2
                            if dist_sq < min_dist_sq:
                                obstacles_clear = False
                                break
                        if not obstacles_clear:
                            break
                    if not obstacles_clear:
                        continue

                obstacle_blocks_path = False
                if self.num_obstacles > 0:
                    dist_robot_goal = np.linalg.norm(robot_pos - goal_pos)
                    for i, o_pos in enumerate(obs_pos):
                        dist_robot_obs = np.linalg.norm(robot_pos - o_pos)
                        dist_goal_obs = np.linalg.norm(goal_pos - o_pos)
                        if (
                            abs(dist_robot_obs + dist_goal_obs - dist_robot_goal)
                            < current_env_obstacle_radii[i] * 2
                        ):
                            obstacle_blocks_path = True
                            break
                    if not obstacle_blocks_path:
                        continue
                else:
                    obstacle_blocks_path = True

                # Pass
                valid_placement = True

            if not valid_placement:
                print(
                    f"Warning: Failed to find valid initial placement for env {env_idx} after {max_placement_attempts} attempts."
                )

            robot_pos_np[env_idx] = robot_pos
            goal_pos_np[env_idx] = goal_pos
            if self.num_obstacles > 0:
                obstacle_positions_np[env_idx] = obs_pos

        # Update only the reset indices in the tensors
        device = self.device
        self._robot_pos[reset_indices] = torch.from_numpy(
            robot_pos_np[reset_indices]
        ).to(device)
        self._goal_pos[reset_indices] = torch.from_numpy(goal_pos_np[reset_indices]).to(
            device
        )
        self._obstacle_positions[reset_indices] = torch.from_numpy(
            obstacle_positions_np[reset_indices]
        ).to(device)
        self._last_velocity[reset_indices] = 0
        self._elapsed_steps[reset_indices] = 0
        self.episode_length_buf[reset_indices] = 0

    def close(self):
        """Cleans up resources, like the rendering window."""
        if self.figure is not None:
            plt.close(self.figure)
            self.figure = None
            self.ax = None
            self.robot_patch = None
            self.goal_patch = None
            self.obstacle_patches = None
            if plt.isinteractive():
                plt.ioff()


# add main function for testing
if __name__ == "__main__":
    env = UnifiedNavigationEnv(
        render_mode="human", device="cpu", num_envs=1
    )  # Specify device, test with 1 env for render
    obs, extras = env.reset()  # Use the VecEnv API return format (obs is tensor)
    # done is now a tensor
    done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    for _ in range(env.max_episode_length * 2):  # Run for longer to see resets
        if done.any():  # Check if any env is done
            print("Resetting environment(s)")
            # Note: In a real scenario, the runner handles resets based on 'done'.
            # Here, we manually break or could call reset again if needed.
            # For simplicity, just break the loop or let it run.
            # obs, extras = env.reset() # Optional: reset manually if not handled externally
            # done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            pass  # Let the loop continue to show post-reset behavior if internal reset happens

        # Generate random actions as torch tensor on the correct device
        # Use torch.rand for tensor-native random generation
        action = (
            torch.rand(env.num_envs, env.num_actions, device=env.device) * 2 - 1
        ) * env.max_velocity
        obs, reward, done, extras = env.step(action)  # Use VecEnv API (all tensors)

        # Rendering still uses numpy arrays from internal state (first env)
        env.render()

        # Optional: Print step info
        # print(f"Step: {extras['elapsed_steps'][0].item()}, Reward: {reward[0].item():.2f}, Done: {done[0].item()}")

        if done.all():  # Exit if all envs are done
            print("All environments finished.")
            break

    env.close()
    for _ in range(env.max_episode_length * 2):  # Run for longer to see resets
        if done.any():  # Check if any env is done
            print("Resetting environment(s)")
            # Note: In a real scenario, the runner handles resets based on 'done'.
            # Here, we manually break or could call reset again if needed.
            # For simplicity, just break the loop or let it run.
            # obs, extras = env.reset() # Optional: reset manually if not handled externally
            # done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            pass  # Let the loop continue to show post-reset behavior if internal reset happens

        # Generate random actions as torch tensor on the correct device
        # Use torch.rand for tensor-native random generation
        action = (
            torch.rand(env.num_envs, env.num_actions, device=env.device) * 2 - 1
        ) * env.max_velocity
        obs, reward, done, extras = env.step(action)  # Use VecEnv API (all tensors)

        # Rendering still uses numpy arrays from internal state (first env)
        env.render()

        # Optional: Print step info
        # print(f"Step: {extras['elapsed_steps'][0].item()}, Reward: {reward[0].item():.2f}, Done: {done[0].item()}")

        if done.all():  # Exit if all envs are done
            print("All environments finished.")
            break

    env.close()
