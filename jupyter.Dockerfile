FROM jupyter/scipy-notebook:python-3.9.5

WORKDIR $HOME
ENV MODULE_PATH=/opt/conda/lib/python3.9/site-packages MY_PKG=task_on_nav_env

RUN git clone https://github.com/jr-robotics/robo-gym.git
ADD ./ ./${MY_PKG}

RUN pip install -e ./robo-gym && pip install pfrl &&\
    cp -r ${MY_PKG}/mir_nav robo-gym/robo_gym/envs/ && \
    cat ${MY_PKG}/env_registration1 >> robo-gym/robo_gym/__init__.py && \
    cat ${MY_PKG}/env_registration2 >> robo-gym/robo_gym/envs/__init__.py && \
    # below maybe unnecessary
    cp ${MY_PKG}/server_modules/robot_server/* ${MODULE_PATH}/robo_gym_server_modules/robot_server/ && \
    cp ${MY_PKG}/server_modules/server_manager/* ${MODULE_PATH}/robo_gym_server_modules/server_manager/
