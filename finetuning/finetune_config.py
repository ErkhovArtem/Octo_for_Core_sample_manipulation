import os

save_dir = "/home/hyperdog/thesis/checkpoints/night/put_in/put_in_the_box_200_eef_augment"
os.makedirs(save_dir, exist_ok=True)

wandb_experiment_name = "put_in_the_box_200_eef_augment"
dataset_name = "core_sample_dataset:1.0.0"
standardize_fn = "select_delta_eef_and_gripper"
# select_joint_angles_and_gripper

# defaul parameters
pretrained_path =  "hf://rail-berkeley/octo-small-1.5" #"/home/hyperdog/octo/checkpoints/..." # 
data_dir = "/home/hyperdog/tensorflow_datasets"
batch_size = 32
freeze_transformer = False

workspace_augment_kwargs = dict(
random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
random_brightness=[0.1],
random_contrast=[0.9, 1.1],
random_saturation=[0.9, 1.1],
random_hue=[0.05],
augment_order=[
    "random_resized_crop",
    "random_brightness",
    "random_contrast",
    "random_saturation",
    "random_hue",
],
)
wrist_augment_kwargs = dict(
random_brightness=[0.1],
random_contrast=[0.9, 1.1],
random_saturation=[0.9, 1.1],
random_hue=[0.05],
augment_order=[
    "random_brightness",
    "random_contrast",
    "random_saturation",
    "random_hue",
],
)