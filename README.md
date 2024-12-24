# Continual Multi-Objective Reinforcement Learning via Reward Model Rehearsal

This repository contains implementation for Continual Multi-Objective Reinforcement Learning via Reward Model Rehearsal (CORe3).

## Environment Installation
Python version: $3.7$

Build the environment by running:

```
pip install -r requirements.txt
```

or

```
conda env create -f environment.yaml
```

Install the Fruit Tree Navigation (FTN) environment by running:

```
pip install -e src/envs/ftn
```

Install the Grid World environment by running:

```
pip install -e src/envs/grid
```

For mujoco benchmark, you should first install mujoco-py of version 2.1 by following the instructions in [mujoco-py](<https://github.com/openai/mujoco-py>). And then install the multi-objectives version by running:
```
pip install -e src/envs/mujoco
```

## Run an experiment

```
python3 src/main.py --config=[Algorithm name] --env-config=[Benchmark name]
```

The config files act as defaults for an algorithm or benchmark. They are all located in `src/config`. `--config` refers to the config files in `src/config/algs` including CORe3, CORe3 (oracle) and Finetune. `--env-config` refers to the config files in `src/config/envs`, including `ftn` as the Fruit Tree Navigation benchmark, `grid` as the Grid World benchmark,  and `mujoco` as the Ant/Hopper benchmark (https://github.com/openai/mujoco-py).

All results will be stored in the `results` folder.

For example, you can run our code as shown bellow:
For discrete control benchmark, run CORe3 on Grid benchmark:
```
python3 src/main.py --config=dqn --env-config=grid --use_world_model=True --oracle=Flase --world_model_type=mlp
```
Run CORe3 (oracle) on Grid benchmark:
```
python3 src/main.py --config=dqn --env-config=grid --use_world_model=True --oracle=True 
```
Run Finetune on FTN benchmark:
```
python3 src/main.py --config=dqn --env-config=ftn --use_world_model=False --oracle=False
```
For continuous control benchmark, you should select the environment config `key` in `src/config/envs/mujoco.yaml` from `MO-Ant-v2` and `MO-Hopper-v2`. For example:
```
env_args:
  key: "MO-Hopper-v2"  
```
And the run CORe3 on Hopper benchmark:
```
python3 src/main.py --config=td3 --env-config=mujoco --use_world_model=True --oracle=False --world_model_type=mlp
```

## Publication

If you find this repository useful, please [cite our paper](https://www.ijcai.org/proceedings/2024/490):

```
@inproceedings{core3,
  title     = {Continual Multi-Objective Reinforcement Learning via Reward Model Rehearsal},
  author    = {Lihe Li and Ruotong Chen and Ziqian Zhang and Zhichao Wu and Yi-Chen Li and Cong Guan and Yang Yu and Lei Yuan},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence},
  pages     = {4434--4442},
  year      = {2024}
}
```