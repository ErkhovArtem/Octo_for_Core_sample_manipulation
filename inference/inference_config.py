import numpy as np

# set up logging
enable_logging = False
experiment_name = "shampoo"

# load model
tasks = ["Pick up a core sample and put it in the box", "Take a core sample out of the box and put it on the red sheet"]
                         
checkpoint_step = 49999
checkpoint_path = "/home/hyperdog/thesis/checkpoints/night/put_in/put_in_the_box_200_eef_augment" # _no_augmentation     

# env configuration
wrist_camera_id = 0
base_pose = np.array([-0.09973652, 
          -1.6148599 ,  
          1.5400181 ,  
          0.1607126 ,  
          1.4716442 ,
        3.1881762 ])

proprio_type = "eef_pose" #"joint_angles"

# np.array([-0.001450840626851857,
#         -1.5725587050067347,
#         1.5713553428649902,
#         0.0008134841918945312,
#         1.5712484121322632,
#         3.142502719560732,])

env_config = {"max_joint_rotation": 0.15, "gripper_opened_pose": 50, "gripper_closed_pose": 155, 
              "action_space": "delta_eef_pose"} # delta_joint_angles