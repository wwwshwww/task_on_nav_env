# task_on_nav_env

## Installation

### Docker

This Environment side can use from docker image below.   
`moyash/robo-gym-env-jupyter:cuberoom`

Usage:

```bash
docker run --gpus all --user=root -p 8888:8888 moyash/robo-gym-env-jupyter:cuberoom jupyter lab --allow-root --LabApp.token='' --ip='0.0.0.0'
```

You can use this environment with jupyter lab by open http://localhost:8888 on your browser.

### Manually

```bash
git clone https://github.com/jr-robotics/robo-gym.git
git clone https://github.com/wwwshwww/task_on_nav_env.git
pip install -e ./robo-gym
cp -r task_on_nav_env/mir_nav robo-gym/robo_gym/envs/
cat task_on_nav_env/env_registration1 >> robo-gym/robo_gym/__init__.py
cat task_on_nav_env/env_registration2 >> robo-gym/robo_gym/envs/__init__.py
```

If using a slower machine, run below as well.
Have to change path that will be placed to paste to appropriate place of `robo_gym_sever_modules` installed.

```bash
# may be unnecessary
cp task_on_nav_env/server_modules/robot_server/* /usr/local/lib/python3.6/dist-packages/robo_gym_server_modules/robot_server/
cp task_on_nav_env/server_modules/server_manager/* /usr/local/lib/python3.6/dist-packages/robo_gym_server_modules/server_manager/
```

## Details

Write you later.

test code ↓

```python
import gym
import robo_gym
from robo_gym.wrappers.exception_handling import ExceptionHandling
import numpy as np

env_id = 'CubeRoomSearchLikeContinuously-v0'
# env_id = 'NoObstacleNavigationMir100Sim-v0'
# target_ip = '10.244.2.247'
target_ip = 'robot-server'

# initialize environment
env = gym.make(env_id, ip=target_ip, gui=True)
env = ExceptionHandling(env)

state = env.reset(new_room=True, new_agent_pose=True)
```

### Robot Server Side

https://github.com/wwwshwww/task_on_nav_robot_server
