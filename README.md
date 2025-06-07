## Project Description

This repository demonstrates the adaptation of the [Octo foundation model](https://github.com/octo-models/octo) for robotic manipulation of geological core samples.

## Implementation details

Training process:
- Finetuning dataset with 200 demonstrations;
- Observation: wrist-mounted camera and third-person view camera;
- Action space: delta end-effector position;
- Policy head: L1 head;

Hardware:

`UR3 manipulator` with `Robotic 2F-85` was used for inference. 

## Code Structure

Repository contains parameters of my fine-tuning process, WandB logs (training curves + metrics), and environment to evaluate model on real UR3 robot.

|                     | File                                                    | Description                                                                   |
|---------------------|---------------------------------------------------------|-------------------------------------------------------------------------------|
| Finetuning Loop     | [finetune.py](finetuning/finetune.py)                   | Main finetuning script.                                                       |
| Finetuning Config   | [funetune_config.py](finetuning/finetune_config.py)     | Defines main hyperparameters for the finetuning run.                          |
| Inference Loop      | [inference.py](inference/inference.py)                  | Main inference script.                                                        |
| Inference Config    | [inference_config.py](inference/inference_config.py)    | Defines main hyperparameters for the inference run.                           |
| Environment         | [env.py](inference/envs/env.py)                         | Custom robot environment                                                      |
