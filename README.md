# task_on_nav_env

## Installation

See both this repository and below repository.

```
git clone https://github.com/wwwshwww/task_on_nav_env.git
cd task_on_nav_env
cp -r mir_nav /usr/local/lib/python3.6/dist-packages/robo_gym/envs/
cat env_registration1 >> /usr/local/lib/python3.6/dist-packages/robo_gym/__init__.py
cat env_registration2 >> /usr/local/lib/python3.6/dist-packages/robo_gym/env/__init__.py
cp server_modules/robot_server/* /usr/local/lib/python3.6/dist-packages/robo_gym_server_modules/robot_server/
cp server_modules/server_manager/* /usr/local/lib/python3.6/dist-packages/robo_gym_server_modules/server_manager/
```

### Robot Server Side

https://github.com/wwwshwww/task_on_nav_robot_server