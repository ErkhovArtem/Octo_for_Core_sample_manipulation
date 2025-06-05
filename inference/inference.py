"""
Main execution script for robot control using OctoModel with force feedback.

Handles:
- Model loading and initialization
- Robot and sensor interfacing
- Real-time control loop
- User interaction and logging
"""

from time import sleep, time
import numpy as np
import keyboard
from octo.model.octo_model import OctoModel
import jax
from inference_config import *
import sys
import os

# Add local environment path to system path
lib_path = os.path.join(os.path.dirname(__file__), 'envs')
sys.path.append(lib_path)

from cameras import RealSenseCamera, WebCamera
from echo_teleoperation import Echo
from ur_rtde import UR3Teleop
from env import EchoEnv
from utils import Logger, NullLogger, TaskManager, construct_observation

def pause(env, logger, task_manager):
    """Pause program execution and handle user input commands.
    
    Returns:
        int: -1 to exit program, 0 to continue execution
    """
    timeout = False
    while True:
        try:
            if keyboard.is_pressed('e'):
                return -1  # Exit signal
            
            if keyboard.is_pressed('b'):
                print('---------------------------------')
                print('Moving to base pose...')
                env.reset()
                logger.reset()
                timeout = True
                
            if keyboard.is_pressed('l'):
                print('---------------------------------')
                print('Saving current episode...')
                logger.save()
                timeout = True
                
            if keyboard.is_pressed('r'):
                print('---------------------------------')
                print('Resuming program execution...')
                return 0  # Continue signal
            
            if keyboard.is_pressed('t'):
                print('---------------------------------')
                print('Switching task...')
                print(f"Model task now is: {task_manager.next_task()}")
                timeout = True        
                
        except Exception as e:
            print(f"Keyboard input error: {e}")
            
        if timeout:
            timeout = False
            sleep(0.1)  # Small delay to prevent rapid repeated triggers

def initialize_hardware():
    """Initialize all hardware interfaces.
    
    Returns:
        tuple: (robot, camera_main, camera_wrist, device) initialized objects
    """
    # Robot initialization
    robot = UR3Teleop(
        ip="192.168.1.110",
        base_pose=base_pose,
        lookahead_time=0.1,
        gain=200
    )
    
    # Camera initialization
    camera_main = RealSenseCamera(
        capture_frequency=30,
        width=640,
        height=480
    )
    camera_wrist = WebCamera(camera_id=wrist_camera_id)
    
    # Force sensor initialization
    device = Echo()
    
    return robot, camera_main, camera_wrist, device

def main():
    """Main execution loop for robot control."""
    # Load pretrained model
    print("Loading OctoModel...")
    global model
    model = OctoModel.load_pretrained(checkpoint_path, checkpoint_step)
    
    # Initialize hardware
    robot, camera_main, camera_wrist, device = initialize_hardware()
    
    # Create environment
    env = EchoEnv(robot, device, camera_main, camera_wrist, env_config)
    
    # Initialize logger
    logger = Logger(experiment_name) if enable_logging else NullLogger()
    
    # Prepare task
    task_manager = TaskManager(tasks)
    # task = model.create_tasks(texts=[tasks[1]])
    
    # Load normalization statistics
    action_proprio_metadata = jax.tree_map(
        lambda x: np.array(x),
        model.dataset_statistics,
        is_leaf=lambda x: isinstance(x, list),
    )
    
    # Reset environment
    observation = env.reset()
    print('Starting teleoperation...')
    
    # Performance monitoring
    start_time = time()
    cycle_count = 0
    
    # Main control loop
    
    while True:
        cycle_count += 1
        
        # Process observation and get action
        observation = construct_observation(
            observation, 
            action_proprio_metadata["proprio"],
            proprio_type
        )
        
        action = model.sample_actions(
            observation, 
            task_manager.get_task(model),
            rng=jax.random.PRNGKey(0),
            unnormalization_statistics=model.dataset_statistics["action"],
        )[0][0]
        
        # Execute action
        observation = env.step(action)
        logger.log(observation)
        
        # Handle user input
        try:
            if keyboard.is_pressed('s'):
                # Calculate and display frequency
                freq = cycle_count / (time() - start_time)
                print(f'Inference frequency: {freq:.2f} Hz')
                print('---------------------------------')
                print('Program paused')
                
                cmd = pause(env, logger, task_manager)
                if cmd == -1:
                    break
                    
                # Reset performance counters
                cycle_count = 0
                start_time = time()
                
        except Exception as e:
            print(f"Control loop error: {e}")
            
    
    # Cleanup resources
    camera_main.release()
    camera_wrist.release()
    print('Exit program...')

if __name__ == "__main__":
    main()