import time
import copy
import jax.numpy as jnp
from PIL import Image
from scipy.spatial.transform import Rotation as R
import numpy as np

from cameras import RealSenseCamera
from utils import quaternion_multiply

class EchoEnv:
    """Environment for executing Octo model inference on a UR3 robot using force data from Echo teleoperation system.
    
    Provides unified interface for:
    - Robot control and state monitoring
    - Multi-camera vision system (scene and wrist views)
    - Force sensor integration
    - Safety checks and error handling
    
    Args:
        robot: Robot control interface
        device: Proprioceptive data interface (force sensor)
        camera_main: Scene-view camera (RGB/RGBD)
        camera_wrist: Wrist-mounted camera (RGB)
        env_config: Dictionary containing configuration:
            - max_joint_rotation: Maximum allowed joint movement (radians)
            - max_force: Maximum allowed contact force (normalized 0-1)
    """
    
    def __init__(self, robot, device, camera_main, camera_wrist, env_config):
        """Initialize hardware interfaces and environment state."""
        self.camera_main = camera_main
        self.camera_wrist = camera_wrist
        self.robot = robot
        self.device = device
        self.previous_state = None
        self.env_config = env_config
        
    def step(self, action=None):
        """Execute one timestep of the environment.
        
        Args:
            action: Optional 7D array containing:
                - 6D joint angle deltas (radians)
                - 1D gripper position delta (normalized -1 to 1)
                
        Returns:
            Dictionary containing stacked observations:
                - image_primary: Scene camera frames (1, 2, H, W, 3)
                - image_wrist: Wrist camera frames (1, 2, H, W, 3) 
                - proprio: Proprioceptive state (1, 2, 2) [gripper_pos, force]
                - Timestep and padding metadata
        """
        if action is not None:
            self._apply_action(action)

        # Get current observations
        image_main = self._get_image(self.camera_main, resize=256)
        image_wrist = self._get_image(self.camera_wrist, resize=128)
        proprio = self._get_proprio()
        
        # Initialize previous state if first step
        if self.previous_state is None:
            self.previous_state = {
                'image_main': copy.copy(image_main),
                'image_wrist': copy.copy(image_wrist),
                'proprio': copy.copy(proprio)
            }

        # Stack current and previous observations
        obs = {
            "image_primary": jnp.stack([self.previous_state['image_main'], image_main], axis=0)[jnp.newaxis, ...],
            "image_wrist": jnp.stack([self.previous_state['image_wrist'], image_wrist], axis=0)[jnp.newaxis, ...],
            "proprio": jnp.stack([self.previous_state['proprio'], proprio], axis=0)[jnp.newaxis, ...],
            "timestep_pad_mask": jnp.array([[True, True]]),
            "timestep": jnp.array([[0, 1]]),
            "pad_mask_dict": {
                "timestep": jnp.array([[True, True]]),
                "image_primary": jnp.array([[True, True]]),
                "image_wrist": jnp.array([[True, True]]),
                "proprio": jnp.array([[True, True]])
            },
            "task_completed": jnp.zeros((1, 2, 4))
        }
        
        # Update previous state
        self.previous_state = {
            'image_main': copy.copy(image_main),
            'image_wrist': copy.copy(image_wrist),
            'proprio': copy.copy(proprio)
        }
        
        return obs
    
    def reset(self):
        """Reset environment to initial state.
        
        Returns:
            Initial observation dictionary (same format as step())
        """
        self.robot.move_to_base_pose()
        time.sleep(1)  # Allow for movement completion
        self.previous_state = None
        return self.step()

    def _get_image(self, camera, resize):
        """Capture and preprocess camera frame.
        
        Args:
            camera: Camera interface object
            resize: Target resolution (square)
            
        Returns:
            jnp.array: Resized RGB image (resize, resize, 3)
        """
        frame = camera.get_frame(depth=False) if isinstance(camera, RealSenseCamera) else camera.get_frame()
        return jnp.array(
            Image.fromarray(frame).resize(
                (resize, resize), 
                Image.Resampling.LANCZOS
            )
        )

    def _apply_action(self, action):
        """Execute robot movement with safety checks.
        
        Args:
            action: 7D action vector
            
        Raises:
            RuntimeError: If joint movement exceeds safety limits
        """

        if self.env_config["action_space"] == "delta_joint_angles":

            if max(abs(action[:6])) > self.env_config['max_joint_rotation']:
                raise RuntimeError(
                    f"Joint movement {max(abs(action[:6]))} "
                    f"exceeds limit {self.env_config['max_joint_rotation']}"
                )

            current_angles = self.robot.get_current_joint_angles()
            target_angles = current_angles + action[:6]
            
            target_gripper_pose = self.env_config["gripper_closed_pose"] if round(action[-1]) else self.env_config["gripper_opened_pose"]
            self.robot.move_to_pose(target_angles, target_gripper_pose)
        
        elif self.env_config["action_space"] == "delta_eef_pose":
            target_tcp_pose = np.zeros(6)
            current_tcp_pose = self.robot.get_current_tcp_pose()
            target_tcp_pose[:3] = current_tcp_pose[:3] + action[:3]

            rot = R.from_rotvec(current_tcp_pose[3:]) * R.from_euler('zxy', action[3:6])
            target_tcp_pose[3:] = rot.as_rotvec()

            target_gripper_pose = self.env_config["gripper_closed_pose"] if round(action[-1]) else self.env_config["gripper_opened_pose"]
            q = self.robot.getInverseKinematics(target_tcp_pose)
            if max(abs(q - self.robot.get_current_joint_angles())) > self.env_config['max_joint_rotation']:
                raise RuntimeError(
                    f"Joint movement {max(abs(action[:6]))} "
                    f"exceeds limit {self.env_config['max_joint_rotation']}"
                )
            self.robot.move_to_pose(q, target_gripper_pose)

        else:
            raise ValueError(f"Invalid action space type: {self.env_config['action_space']}")
        


    def _get_proprio(self):
        """Read current proprioceptive state.
        
        Returns:
            jnp.array: [6x current joint angles, 6x current tcp pose, gripper position (0-1)]
            
        """
        
        gripper_pose = self.robot.get_current_gripper_pose()[0]
        
        return jnp.array([
            *self.robot.get_current_joint_angles(), *self.robot.get_current_tcp_pose(),
            self._binarize_gripper_pose(gripper_pose)
        ])
    
    def _binarize_gripper_pose(self, gripper_pose):

        return gripper_pose > (self.env_config["gripper_closed_pose"] + self.env_config["gripper_opened_pose"])/2