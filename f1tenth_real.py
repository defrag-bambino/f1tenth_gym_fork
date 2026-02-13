"""
F1Tenth Real Car Environment Wrapper for DreamerV3.

This module provides a ROS2-based environment that matches the simulation
interface, enabling seamless sim-to-real transfer for DreamerV3 training.

Usage:
    from f1tenth_real import F1TenthReal
    
    env = F1TenthReal(
        scan_beams=32,
        max_speed=5.0,  # Conservative for safety
    )
    
    # Same interface as F1Tenth simulation
    obs = env.step({'reset': True})
    obs = env.step({'action': np.array([0.1, 2.0])})  # [steering, speed]
"""

import functools
import time
import threading
import queue
from typing import Optional, Dict, Any
import numpy as np

import elements
import embodied


class F1TenthReal(embodied.Env):
    """
    ROS2-based real car environment for F1tenth with DreamerV3 interface.
    
    This environment provides the same interface as the F1Tenth simulation
    wrapper, enabling seamless transfer between simulation and real hardware.
    
    Observation space matches simulation:
        - scan: LiDAR scan (subsampled)
        - linear_vel_x: Forward velocity
        - ang_vel_z: Angular velocity
        - delta: Current steering angle
        
    Action space matches simulation:
        - action: [steering_angle, speed]
    
    The environment handles:
        - ROS2 communication with the car
        - Real-time step synchronization
        - Collision detection from LiDAR
        - Velocity-based reward similar to simulation
    """

    def __init__(
        self,
        scan_beams: int = 32,
        scan_original_size: int = 1080,
        max_speed: float = 5.0,
        max_steering_angle: float = 0.4189,
        step_frequency: float = 20.0,
        collision_threshold: float = 0.3,
        collision_penalty: float = -10.0,
        scan_topic: str = '/scan',
        odom_topic: str = '/odom',
        drive_topic: str = '/drive',
        obs_features: Optional[list] = None,
        use_sliding_window: bool = True,
        window_size: int = 3,
        manual_reset: bool = True,
        reset_timeout: float = 30.0,
        velocity_reward_scale: float = 1.0,
        dist_to_wall_start_neg_rew: float = 0.3,
        **kwargs
    ):
        """
        Initialize the real car environment.
        
        Args:
            scan_beams: Number of LiDAR beams to use (subsampled from scan_original_size)
            scan_original_size: Original LiDAR scan size (typically 1080)
            max_speed: Maximum allowed speed (m/s) - safety limit
            max_steering_angle: Maximum steering angle (rad)
            step_frequency: Environment step frequency (Hz)
            collision_threshold: Minimum distance to consider collision (m)
            collision_penalty: Reward penalty for collision
            scan_topic: ROS2 topic for LiDAR scan
            odom_topic: ROS2 topic for odometry
            drive_topic: ROS2 topic for drive commands
            obs_features: List of observation features (for compatibility)
            use_sliding_window: Whether to smooth observations
            window_size: Sliding window size for observation smoothing
            manual_reset: Whether reset requires manual intervention
            reset_timeout: Timeout waiting for manual reset (seconds)
            velocity_reward_scale: Scale factor for velocity reward
            dist_to_wall_start_neg_rew: Distance to wall (m) where negative reward starts (default: 0.4)
        """
        # Store configuration
        self._scan_beams = scan_beams
        self._scan_original_size = scan_original_size
        self._max_speed = max_speed
        self._max_steering_angle = max_steering_angle
        self._step_frequency = step_frequency
        self._step_period = 1.0 / step_frequency
        self._collision_threshold = collision_threshold
        self._collision_penalty = collision_penalty
        self._scan_topic = scan_topic
        self._odom_topic = odom_topic
        self._drive_topic = drive_topic
        self._use_sliding_window = use_sliding_window
        self._window_size = window_size
        self._manual_reset = manual_reset
        self._reset_timeout = reset_timeout
        self._velocity_reward_scale = velocity_reward_scale
        self._dist_to_wall_start_neg_rew = dist_to_wall_start_neg_rew
        
        # Calculate scan subsampling indices
        self._scan_indices = np.linspace(
            0, scan_original_size - 1, scan_beams, dtype=np.int32
        )
        
        # State variables
        self._done = True
        self._info = {}
        self._last_step_time = 0.0
        self._current_steering = 0.0
        self._episode_steps = 0
        self._total_steps = 0
        
        # Current sensor readings (protected by lock)
        self._lock = threading.Lock()
        self._current_scan = None
        self._current_odom = None
        self._data_ready = threading.Event()
        
        # Observation history for sliding window
        from collections import deque
        self._obs_history = deque(maxlen=window_size)
        
        # ROS2 initialization flag
        self._ros_initialized = False
        self._ros_node = None
        
        # Initialize ROS2
        self._init_ros2()

    def _init_ros2(self):
        """Initialize ROS2 node and subscriptions."""
        try:
            import rclpy
            from rclpy.node import Node
            from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
            from sensor_msgs.msg import LaserScan
            from nav_msgs.msg import Odometry
            from ackermann_msgs.msg import AckermannDriveStamped
            from transforms3d.euler import quat2euler
        except ImportError as e:
            raise ImportError(
                "ROS2 dependencies not found. Please install rclpy, sensor_msgs, "
                "nav_msgs, ackermann_msgs, and transforms3d.\n"
                "Make sure you have sourced your ROS2 workspace.\n"
                f"Original error: {e}"
            )
        
        # Store message types and conversion function
        self._LaserScan = LaserScan
        self._Odometry = Odometry
        self._AckermannDriveStamped = AckermannDriveStamped
        self._quat2euler = quat2euler
        
        # Initialize ROS2 if not already done
        if not rclpy.ok():
            rclpy.init()
        
        # Create node
        self._ros_node = rclpy.create_node('f1tenth_dreamer_env')
        
        # QoS profile for sensor data
        # Use RELIABLE for consistency (test script uses default which is RELIABLE)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Subscribers
        self._scan_sub = self._ros_node.create_subscription(
            LaserScan,
            self._scan_topic,
            self._scan_callback,
            sensor_qos
        )
        
        self._odom_sub = self._ros_node.create_subscription(
            Odometry,
            self._odom_topic,
            self._odom_callback,
            10
        )
        
        # Publisher
        self._drive_pub = self._ros_node.create_publisher(
            AckermannDriveStamped,
            self._drive_topic,
            10
        )
        
        self._ros_initialized = True
        print(f"[F1TenthReal] ROS2 initialized - listening on {self._scan_topic}, {self._odom_topic}")
        
        # Start ROS2 spinner in background thread
        self._spin_thread = threading.Thread(target=self._ros_spin, daemon=True)
        self._spin_thread.start()
        
        print(f"[F1TenthReal] Waiting for initial sensor data...")
        
        # Wait a bit for initial data
        import time
        time.sleep(2.0)
        
        # Check if we got any data
        with self._lock:
            if self._current_scan is None:
                print(f"[F1TenthReal] WARNING: No data on {self._scan_topic} yet!")
                print(f"[F1TenthReal] Check: ros2 topic list | grep {self._scan_topic}")
                print(f"[F1TenthReal] Check: ros2 topic hz {self._scan_topic}")
            if self._current_odom is None:
                print(f"[F1TenthReal] WARNING: No data on {self._odom_topic} yet!")
                print(f"[F1TenthReal] Check: ros2 topic list | grep {self._odom_topic}")
                print(f"[F1TenthReal] Check: ros2 topic hz {self._odom_topic}")

    def _ros_spin(self):
        """Background thread for ROS2 spinning."""
        import rclpy
        print("[F1TenthReal] ROS2 spinner thread started")
        while rclpy.ok() and self._ros_initialized:
            try:
                rclpy.spin_once(self._ros_node, timeout_sec=0.1)
            except Exception as e:
                print(f"[F1TenthReal] Spinner error: {e}")
                break
        print("[F1TenthReal] ROS2 spinner thread stopped")

    def _scan_callback(self, msg):
        """Process incoming LiDAR scan."""
        # First callback - log that we got data
        if self._current_scan is None:
            print(f"[F1TenthReal] First scan received! ({len(msg.ranges)} beams)")
        
        scan_array = np.array(msg.ranges, dtype=np.float32)
        
        # Handle inf and nan values
        scan_array = np.nan_to_num(scan_array, nan=30.0, posinf=30.0, neginf=0.0)
        
        # Subsample or resample to desired size
        if len(scan_array) == self._scan_original_size:
            subsampled = scan_array[self._scan_indices]
        elif len(scan_array) == self._scan_beams:
            subsampled = scan_array
        else:
            # Resample to desired size
            indices = np.linspace(0, len(scan_array) - 1, self._scan_beams, dtype=int)
            subsampled = scan_array[indices]
        
        # Clip to reasonable range
        subsampled = np.clip(subsampled, 0.0, 30.0)
        
        with self._lock:
            self._current_scan = subsampled
            if self._current_odom is not None:
                self._data_ready.set()

    def _odom_callback(self, msg):
        """Process incoming odometry."""
        # First callback - log that we got data
        if self._current_odom is None:
            print(f"[F1TenthReal] First odom received!")
        
        # Extract velocities
        linear_vel_x = abs(msg.twist.twist.linear.x)
        ang_vel_z = msg.twist.twist.angular.z
        
        # Extract pose
        pose_x = msg.pose.pose.position.x
        pose_y = msg.pose.pose.position.y
        quat = msg.pose.pose.orientation
        _, _, pose_theta = self._quat2euler([quat.w, quat.x, quat.y, quat.z])
        
        with self._lock:
            self._current_odom = {
                'linear_vel_x': np.float32(linear_vel_x),
                'ang_vel_z': np.float32(ang_vel_z),
                'pose_x': np.float32(pose_x),
                'pose_y': np.float32(pose_y),
                'pose_theta': np.float32(pose_theta),
            }
            if self._current_scan is not None:
                self._data_ready.set()

    def _publish_drive(self, steering: float, speed: float):
        """Publish drive command to the car."""
        # Apply safety limits
        steering = np.clip(steering, -self._max_steering_angle, self._max_steering_angle)
        speed = np.clip(speed, -0.3, self._max_speed)  # Allow slight reverse
        
        msg = self._AckermannDriveStamped()
        msg.header.stamp = self._ros_node.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.steering_angle = float(steering)
        msg.drive.speed = float(speed)
        
        self._drive_pub.publish(msg)
        
        # Track current steering for observation
        self._current_steering = steering

    def _stop_car(self):
        """Send stop command to the car."""
        self._publish_drive(0.0, 0.0)

    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation from sensors."""
        # Wait for data with timeout
        if not self._data_ready.wait(timeout=2.0):
            print("[F1TenthReal] Warning: Timeout waiting for sensor data")
            # Return safe default values (far distances) if no data
            # Use max range to avoid false collision detection
            return {
                'scan': np.full(self._scan_beams, 30.0, dtype=np.float32),  # Max range, not zeros!
                'linear_vel_x': np.float32(0.0),
                'ang_vel_z': np.float32(0.0),
                'delta': np.float32(0.0),
            }
        
        with self._lock:
            # Clip delta to ensure it's within bounds (floating point precision issues)
            delta_clipped = np.clip(
                self._current_steering,
                -self._max_steering_angle,
                self._max_steering_angle
            )
            # Clamp linear_vel_x to 0 if below 0.5
            linear_vel_x_raw = self._current_odom['linear_vel_x']
            linear_vel_x_clamped = 0.0 if linear_vel_x_raw < 0.5 else linear_vel_x_raw
            obs = {
                'scan': self._current_scan.copy(),
                'linear_vel_x': np.float32(linear_vel_x_clamped),
                'ang_vel_z': self._current_odom['ang_vel_z'],
                'delta': np.float32(delta_clipped),
            }
            self._data_ready.clear()
        
        # Apply sliding window smoothing if enabled
        if self._use_sliding_window and len(self._obs_history) > 0:
            self._obs_history.append(obs)
            smoothed_obs = {}
            for key in obs.keys():
                smoothed_obs[key] = np.mean(
                    [o[key] for o in self._obs_history], axis=0
                ).astype(obs[key].dtype)
            return smoothed_obs
        else:
            self._obs_history.append(obs)
            return obs

    def _check_collision(self, scan: np.ndarray) -> bool:
        """Check if any scan reading indicates collision."""
        # Filter out invalid readings (0.0 or very close to 0)
        # These often indicate no data rather than actual collision
        valid_readings = scan[scan > 0.05]  # Ignore readings below 5cm
        if len(valid_readings) == 0:
            return False  # No valid data, assume no collision
        return np.any(valid_readings < self._collision_threshold)

    def _compute_reward(self, obs: Dict[str, Any], collision: bool) -> float:
        """Compute reward based on velocity, collision status, and distance to walls."""
        if collision:
            return self._collision_penalty
        
        # Velocity-based reward (similar to simulation)
        if obs['linear_vel_x'] > 0.5: # only reward if velocity is greater than 0.5 m/s, because otherwise it abuses odometry error
            velocity_reward = obs['linear_vel_x'] * self._velocity_reward_scale
        else:
            velocity_reward = 0.0
        
        # Distance-based penalty: negative reward when close to walls
        # Find minimum distance to obstacles from scan
        scan = obs.get('scan', None)
        if scan is not None:
            # Filter out invalid readings (inf, nan, or very large values)
            valid_readings = scan[np.isfinite(scan) & (scan < 100.0)]
            if len(valid_readings) > 0:
                min_distance = np.min(valid_readings)
                
                # If distance is less than threshold, apply linear penalty
                if min_distance < self._dist_to_wall_start_neg_rew:
                    # Calculate penalty: 0 at threshold, -100 at collision_threshold
                    # Linear interpolation
                    distance_into_penalty_zone = self._dist_to_wall_start_neg_rew - min_distance
                    penalty_zone_width = self._dist_to_wall_start_neg_rew - self._collision_threshold
                    
                    if penalty_zone_width > 0:
                        # Normalize distance into penalty zone [0, 1]
                        normalized_distance = min(1.0, distance_into_penalty_zone / penalty_zone_width)
                        # Linear penalty from 0 to collision_penalty
                        distance_penalty = self._collision_penalty * normalized_distance
                        velocity_reward *= distance_penalty
        
        #print("total reward: ", velocity_reward)
        return velocity_reward

    @functools.cached_property
    def obs_space(self):
        """Define observation space matching simulation interface."""
        # Add small epsilon to delta bounds for floating point tolerance
        delta_eps = 1e-5
        spaces = {
            'scan': elements.Space(np.float32, (self._scan_beams,), -np.inf, np.inf),
            'linear_vel_x': elements.Space(np.float32, (), -10.0, 30.0),
            'ang_vel_z': elements.Space(np.float32, (), -10.0, 10.0),
            'delta': elements.Space(
                np.float32, (),
                -self._max_steering_angle - delta_eps,
                self._max_steering_angle + delta_eps
            ),
            # Standard DreamerV3 fields
            'reward': elements.Space(np.float32),
            'is_first': elements.Space(bool),
            'is_last': elements.Space(bool),
            'is_terminal': elements.Space(bool),
        }
        return spaces

    @functools.cached_property
    def act_space(self):
        """Define action space matching simulation interface."""
        spaces = {
            # Action: [steering_angle, speed]
            'action': elements.Space(
                np.float32,
                (2,),
                np.array([-self._max_steering_angle, -1.0]),
                np.array([self._max_steering_angle, self._max_speed])
            ),
            'reset': elements.Space(bool),
        }
        return spaces

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute one step in the environment.
        
        Args:
            action: Dictionary with 'action' key containing [steering, speed]
                   or 'reset' key to reset the environment
        
        Returns:
            Observation dictionary with sensor data and metadata
        """
        # Handle reset
        if action.get('reset', False) or self._done:
            return self._reset()
        
        # Extract action
        act = action['action']
        steering = float(act[0])
        speed = float(act[1])
        
        # Rate limiting - ensure consistent step frequency
        current_time = time.time()
        elapsed = current_time - self._last_step_time
        if elapsed < self._step_period:
            time.sleep(self._step_period - elapsed)
        self._last_step_time = time.time()
        
        # Publish drive command
        self._publish_drive(steering, speed)
        
        # Get observation
        obs = self._get_observation()
        
        # Check collision
        collision = self._check_collision(obs['scan'])
        
        # Compute reward
        reward = self._compute_reward(obs, collision)
        
        # Update episode tracking
        self._episode_steps += 1
        self._total_steps += 1
        
        # Determine if episode is done
        done = collision  # End episode on collision
        self._done = done
        
        if done:
            self._stop_car()
            print(f"[F1TenthReal] Episode ended after {self._episode_steps} steps "
                  f"(collision={collision})")
        
        # Build observation dict
        return self._build_obs(obs, reward, is_first=False, is_last=done, is_terminal=collision)

    def _reset(self) -> Dict[str, Any]:
        """Reset the environment for a new episode."""
        self._stop_car()
        self._done = False
        self._episode_steps = 0
        self._current_steering = 0.0
        self._obs_history.clear()
        
        if self._manual_reset:
            print("\n" + "=" * 60)
            print("[F1TenthReal] RESET REQUIRED")
            print("Please place the car at the starting position.")
            print("Press ENTER when ready to continue...")
            print("=" * 60 + "\n")
            
            # Wait for user input with timeout
            import sys
            import select
            
            start_time = time.time()
            while time.time() - start_time < self._reset_timeout:
                # Non-blocking input check (Unix-specific)
                try:
                    if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                        sys.stdin.readline()
                        break
                except:
                    # Fallback for non-Unix systems
                    time.sleep(0.1)
                    # Could add a different input mechanism here
            else:
                print(f"[F1TenthReal] Reset timeout after {self._reset_timeout}s, continuing anyway")
        
        # Wait for fresh sensor data
        self._data_ready.clear()
        if not self._data_ready.wait(timeout=2.0):
            print("[F1TenthReal] Warning: No sensor data available after reset")
        
        # Get initial observation
        obs = self._get_observation()
        self._last_step_time = time.time()
        
        print(f"[F1TenthReal] Episode started (total steps so far: {self._total_steps})")
        
        return self._build_obs(obs, 0.0, is_first=True, is_last=False, is_terminal=False)

    def _build_obs(
        self,
        obs: Dict[str, Any],
        reward: float,
        is_first: bool = False,
        is_last: bool = False,
        is_terminal: bool = False
    ) -> Dict[str, Any]:
        """Build observation dict in DreamerV3 format."""
        result = {
            'scan': obs['scan'],
            'linear_vel_x': obs['linear_vel_x'],
            'ang_vel_z': obs['ang_vel_z'],
            'delta': obs['delta'],
            'reward': np.float32(reward),
            'is_first': is_first,
            'is_last': is_last,
            'is_terminal': is_terminal,
        }
        return result

    def render(self):
        """Render is not applicable for real car."""
        pass

    def close(self):
        """Clean up ROS2 resources."""
        # Prevent double-shutdown
        if not self._ros_initialized:
            return
        
        print("[F1TenthReal] Shutting down...")
        
        try:
            self._stop_car()
        except Exception as e:
            print(f"[F1TenthReal] Warning during stop_car: {e}")
        
        self._ros_initialized = False
        
        # Destroy node first
        if self._ros_node is not None:
            try:
                self._ros_node.destroy_node()
                self._ros_node = None
            except Exception as e:
                print(f"[F1TenthReal] Warning during node destruction: {e}")
        
        # Don't shutdown rclpy - it may be used by other nodes
        # and embodied framework handles shutdown
        # Shutting down here causes "terminate called without an active exception"
        
        print("[F1TenthReal] Shutdown complete")

    @property
    def info(self):
        """Return current info dict."""
        return self._info


class F1TenthRealWithAutomaticReset(F1TenthReal):
    """
    F1Tenth real car environment with automatic reset capability.
    
    This variant uses external position tracking (e.g., Vicon, OptiTrack)
    to automatically detect when the car should be reset and can
    optionally drive back to a starting position.
    
    Requires additional ROS2 topics for external pose tracking.
    """
    
    def __init__(
        self,
        pose_topic: str = '/vrpn_client_node/car/pose',
        start_pose: tuple = (0.0, 0.0, 0.0),  # x, y, theta
        start_tolerance: float = 0.5,
        **kwargs
    ):
        """
        Initialize with automatic reset capability.
        
        Args:
            pose_topic: ROS2 topic for external pose tracking
            start_pose: Starting position (x, y, theta) for automatic reset detection
            start_tolerance: Tolerance for considering car at start position
            **kwargs: Additional arguments passed to F1TenthReal
        """
        # Override manual_reset
        kwargs['manual_reset'] = False
        super().__init__(**kwargs)
        
        self._pose_topic = pose_topic
        self._start_pose = np.array(start_pose, dtype=np.float32)
        self._start_tolerance = start_tolerance
        self._external_pose = None
        
        # Subscribe to external pose if available
        self._init_pose_tracking()
    
    def _init_pose_tracking(self):
        """Initialize external pose tracking subscription."""
        try:
            from geometry_msgs.msg import PoseStamped
            
            self._pose_sub = self._ros_node.create_subscription(
                PoseStamped,
                self._pose_topic,
                self._external_pose_callback,
                10
            )
            print(f"[F1TenthReal] External pose tracking on {self._pose_topic}")
        except Exception as e:
            print(f"[F1TenthReal] Warning: Could not init pose tracking: {e}")
    
    def _external_pose_callback(self, msg):
        """Process external pose updates."""
        with self._lock:
            self._external_pose = np.array([
                msg.pose.position.x,
                msg.pose.position.y,
            ], dtype=np.float32)
    
    def _is_at_start(self) -> bool:
        """Check if car is at starting position."""
        if self._external_pose is None:
            return True  # Assume at start if no tracking
        
        dist = np.linalg.norm(self._external_pose - self._start_pose[:2])
        return dist < self._start_tolerance


def make_real_env(
    scan_beams: int = 32,
    max_speed: float = 5.0,
    automatic_reset: bool = False,
    **kwargs
) -> embodied.Env:
    """
    Factory function to create a real car environment.
    
    Args:
        scan_beams: Number of LiDAR beams
        max_speed: Maximum allowed speed
        automatic_reset: Whether to use automatic reset (requires external tracking)
        **kwargs: Additional arguments passed to environment
    
    Returns:
        F1TenthReal environment instance
    """
    if automatic_reset:
        return F1TenthRealWithAutomaticReset(
            scan_beams=scan_beams,
            max_speed=max_speed,
            **kwargs
        )
    else:
        return F1TenthReal(
            scan_beams=scan_beams,
            max_speed=max_speed,
            **kwargs
        )
