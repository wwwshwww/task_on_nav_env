FROM jupyter/scipy-notebook:python-3.9.5

WORKDIR $HOME
ENV LOCATION /opt/conda/lib/python3.9/site-packages

RUN pip install pfrl

RUN git clone https://github.com/jr-robotics/robo-gym.git && \
    git clone https://github.com/wwwshwww/task_on_nav_env.git && \
    pip install -e ./robo-gym && \
    cp -r task_on_nav_env/mir_nav robo-gym/robo_gym/envs/ && \
    cat task_on_nav_env/env_registration1 >> robo-gym/robo_gym/__init__.py && \
    cat task_on_nav_env/env_registration2 >> robo-gym/robo_gym/envs/__init__.py

RUN cp task_on_nav_env/server_modules/robot_server/* $LOCATION/robo_gym_server_modules/robot_server/ && \
    cp task_on_nav_env/server_modules/server_manager/* $LOCATION/robo_gym_server_modules/server_manager/
