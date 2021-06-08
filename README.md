# task_on_nav_env

## Installation

See both this repository and below repository.

```
git clone https://github.com/jr-robotics/robo-gym.git
git clone https://github.com/wwwshwww/task_on_nav_env.git
pip install -e ./robo-gym
cp -r task_on_nav_env/mir_nav robo-gym/robo_gym/envs/
cat task_on_nav_env/env_registration1 >> robo-gym/robo_gym/__init__.py
cat task_on_nav_env/env_registration2 >> robo-gym/robo_gym/envs/__init__.py
# may be unnecessary
cp task_on_nav_env/server_modules/robot_server/* /usr/local/lib/python3.6/dist-packages/robo_gym_server_modules/robot_server/
cp task_on_nav_env/server_modules/server_manager/* /usr/local/lib/python3.6/dist-packages/robo_gym_server_modules/server_manager/
```

### Robot Server Side

https://github.com/wwwshwww/task_on_nav_robot_server