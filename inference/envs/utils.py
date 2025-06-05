import numpy as np
from pathlib import Path
import jax.numpy as jnp

def normalize(data, metadata):
    """Normalize data using dataset statistics while respecting a mask.
    
    Args:
        data: Input array to normalize
        metadata: Dictionary containing:
            - 'mean': Mean values for normalization
            - 'std': Standard deviation values
            - 'mask' (optional): Boolean mask indicating which elements to normalize
            
    Returns:
        Normalized array where mask is True, original values where False
    """
    mask = metadata.get("mask", np.ones_like(metadata["mean"], dtype=bool))
    return np.where(
        mask,
        (data - metadata["mean"]) / (metadata["std"] + 1e-8),  # Small epsilon to avoid division by zero
        data,
    )

def construct_observation(observation, metadata, proprio_type):
    """Process and normalize observation data based on configuration.
    
    Args:
        observation: Raw observation dictionary containing:
            - 'proprio': Proprioceptive data array
        metadata: Normalization parameters (see normalize())
            
    Returns:
        Processed observation dictionary with normalized proprioceptive data
    """
    # Select relevant proprioceptive data
    if proprio_type == "joint_angles":
        observation['proprio'] = jnp.concatenate([observation['proprio'][..., :6], observation['proprio'][..., -1][..., None]], axis = -1)
    elif proprio_type == "eef_pose":
        observation['proprio'] = jnp.concatenate([observation['proprio'][..., 6:12], observation['proprio'][..., -1][..., None]], axis = -1)
    else:
        raise ValueError(f"Invalid proprio type: {proprio_type}")


    # Normalize selected proprioceptive data
    proprio_norm = normalize(observation['proprio'][0], metadata)
    observation['proprio'] = observation['proprio'].at[0].set(proprio_norm)
    return observation

class Logger:
    """Logs proprioceptive data to disk with automatic episode numbering."""
    
    def __init__(self, experiment_name, proprio_data):
        """Initialize logger with empty buffer and create output directory.
        
        Args:
            experiment_name: Name of experiment (used for subdirectory)
            proprio_data: Type of proprioceptive data being logged
        """
        self.log_data = np.zeros((0, 2))  # Buffer for current episode data
        self.log_dir = Path() / "logs" / experiment_name / proprio_data
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log(self, observation):
        """Append current timestep's proprioceptive data to buffer.
        
        Args:
            observation: Dictionary containing 'proprio' array with shape (1, 2, N)
        """
        data = observation['proprio'][0, 1, -2:][None, ...]  # Extract current timestep
        self.log_data = np.concatenate([self.log_data, data])

    def reset(self):
        """Clear current episode buffer."""
        self.log_data = np.zeros((0, 2))

    def save(self):
        """Save current episode data to disk with auto-incremented filename."""
        if len(self.log_data) == 0:
            return  # Skip saving if no data

        # Find next available episode number
        episode_number = 1
        while (self.log_dir / f"episode_{episode_number}.npy").exists():
            episode_number += 1

        # Save and reset buffer
        np.save(self.log_dir / f'episode_{episode_number}.npy', self.log_data)
        self.reset()

class NullLogger:
    """Dummy logger that implements the Logger interface but does nothing."""
    
    def __init__(self, *args, **kwargs):
        """Accept any arguments but take no action."""
        pass

    def log(self, *args, **kwargs):
        """Accept logging calls but take no action."""
        pass

    def save(self, *args, **kwargs):
        """Accept save calls but take no action."""
        pass

    def reset(self, *args, **kwargs):
        """Accept save calls but take no action."""
        pass

class TaskManager:
    def __init__(self, tasks):
        self.tasks = tasks
        self.current_task_id = 0
        self.current_task = None
    
    def get_task(self, model):

        if self.current_task is None:
            self.current_task = model.create_tasks(texts=[self.tasks[self.current_task_id]])
    
        return self.current_task

    def next_task(self):

        self.current_task_id = (self.current_task_id + 1) % len(self.tasks)
        self.current_task = None
        return self.tasks[self.current_task_id]
    
def quaternion_multiply(quaternion1, quaternion0, reverse = False):
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1

    return np.array([x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
                     -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0], dtype=np.float64)