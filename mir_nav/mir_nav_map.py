import numpy as np
from gym import spaces
from robo_gym.envs.simulation_wrapper import Simulation
from collections import deque

from .utils import transform_2d, make_subjective_image
from .mir_nav import Mir100NavEnv
from .mir_nav import REWARD_DEFAULT, REWARD_MOVE, REWARD_DISCOVER

class CubeRoomEnvObsMapOnly(Mir100NavEnv):
    '''
    Observation: Subjective Occupancy Grid Map: (map_size, map_size,)
    Action: Relative Goal Pose [polar r, polar theta, yaw angle]: (3,)
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_map = None
    
    def _get_observation_space(self):
        occupancy_grid_space = spaces.Box(low=-1, high=100, shape=(self.map_size,self.map_size,), dtype=np.float32)
        return occupancy_grid_space
    
    def _robot_server_state_to_env_state(self, rs_state):
        pose = self.rss_manager.get_from_rs_state(rs_state, 'agent_pose')
        data = np.array(self.rss_manager.get_from_rs_state(rs_state, 'map_data'))
        map_img = data.reshape([self.map_size, self.map_size]).T
        self.original_map = map_img
        odom_x, odom_y, yaw = transform_2d(pose[0], pose[1], pose[2], *self.start_frame)
        pix_x = odom_x / self.resolution
        pix_y = odom_y / self.resolution
        return make_subjective_image(map_img, pix_x, pix_y, yaw).astype(np.float32)
    
class CubeRoomWithTargetFind(CubeRoomEnvObsMapOnly):
    
    def __init__(self, *args, **kwargs):
        Mir100NavEnv.__init__(self, *args, **kwargs)

        self.target_found = [] # flag that target have been found for each

    def reset(self, *args, **kwargs):
        state = super().reset(*args, **kwargs)
        self.target_found = np.full([self.target_num], False)
        return state

    def check_found_new_one(self, threshold) -> bool:
        # check if there is target found
        diff_targets = np.full([self.target_num,2], self.agent_pose[:2]) - self.target_pose[:,:2]
        dist_targets = np.linalg.norm(diff_targets, axis=1)
        are_found = np.logical_and(dist_targets <= threshold, np.logical_not(self.target_found))
        idx_found = np.arange(self.target_num)[are_found]

        found_id = -1
        found = False

        if len(idx_found) > 0:
            found = True
            found_id = idx_found[0]
            self.target_found[found_id] = True

        return found, found_id

class CubeRoomWithMapDifferenceCalculate(CubeRoomEnvObsMapOnly):
    
    def __init__(self, *args, **kwargs):
        Mir100NavEnv.__init__(self, *args, **kwargs)

        self.map_queue = deque(maxlen=2)

    def reset(self, *args, **kwargs):
        state = super().reset(*args, **kwargs)
        self.map_queue = deque(maxlen=2)
        return state

    def step(self, action):
        self.map_queue.append(self.original_map)
        state, reward, done, info = super().step(action)
        return state, reward, done, info

    def calculate_both_maps_diff(self) -> int:
        '''
        Return how much decreased the unknown area as pixel num.
        This function is supposed to be called in the reward function.
        '''
        return np.sum(self.map_queue[-1] < 0) - np.sum(self.original_map < 0)
    
class CubeSearchInCubeRoomObsMapOnly(CubeRoomWithTargetFind, Simulation):
    wait_for_current_action = 5
    found_thresh = 0.8
    move_distance_thresh = 0.7
    
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, gazebo_gui=False, **kwargs):
        opt_wait_moved = 'wait_moved:=false'
        opt_sleep_time = f'sleep_time:={self.wait_for_current_action}'
        opt_gazebo_gui = f'gazebo_gui:={"true" if gazebo_gui else "false"}'
        self.cmd = f"roslaunch task_on_nav_robot_server sim_robot_server.launch {opt_wait_moved} {opt_sleep_time} {opt_gazebo_gui}"

        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        CubeRoomWithTargetFind.__init__(self, rs_address=self.robot_server_ip, **kwargs)
        
    def _reward(self, rs_state, action):
        reward = REWARD_DEFAULT
        done = False
        info = {}
        
        is_found, _ = self.check_found_new_one(threshold=self.found_thresh)
        
        if is_found:
            reward += REWARD_DISCOVER

        if self.move_distance > self.move_distance_thresh:
            reward += REWARD_MOVE
            
        if np.sum(self.target_found) == self.target_num:
            done = True
            info['final_status'] = 'success'
            
        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info['final_status'] = 'max_steps_exceeded'

        return reward, done, info
    
class MapExploreInCubeRoomObsMapOnly(CubeRoomWithMapDifferenceCalculate, Simulation):
    wait_for_current_action = 5
    
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, gazebo_gui=False, **kwargs):
        opt_wait_moved = 'wait_moved:=false'
        opt_sleep_time = f'sleep_time:={self.wait_for_current_action}'
        opt_gazebo_gui = f'gazebo_gui:={"true" if gazebo_gui else "false"}'
        self.cmd = f"roslaunch task_on_nav_robot_server sim_robot_server.launch {opt_wait_moved} {opt_sleep_time} {opt_gazebo_gui}"

        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        CubeRoomWithMapDifferenceCalculate.__init__(self, rs_address=self.robot_server_ip, **kwargs)
        
    def _reward(self, rs_state, action):
        reward = REWARD_DEFAULT
        done = False
        info = {}
        
        explored_pixel_num = self.calculate_both_maps_diff()

        reward += explored_pixel_num

        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info['final_status'] = 'max_steps_exceeded'

        return reward, done, info