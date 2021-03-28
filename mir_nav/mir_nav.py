import sys, time, math, copy
import numpy as np
import gym
from gym import spaces, logger
from gym.utils import seeding

from robo_gym.utils import utils
from robo_gym.utils.exceptions import InvalidStateError, RobotServerError
import robo_gym_server_modules.robot_server.client as rs_client
from robo_gym.envs.simulation_wrapper import Simulation
from robo_gym_server_modules.robot_server.grpc_msgs.python import robot_server_pb2

from .utils import transform_2d, cartesian_to_polar_2d, polar_to_cartesian_2d, relative_to_origin

from typing import List

class Mir100NavEnv(gym.Env):
    real_robot = False
    slam_map_size = 512
    slam_resolution = 0.05
    map_size = 128
    resolution = slam_resolution * (slam_map_size / map_size)
    
    def __init__(self, rs_address=None, max_episode_steps=500, **kwargs):
        self.max_episode_steps = max_episode_steps
        self.elapsed_steps = 0
        
        self.observation_space = self._get_observation_space()
        
#         self.action_space = spaces.Dict({
#             'polar_r': spaces.Box(low=0, high=1, shape=(2,)),
#             'polar_theta': spaces.Box(low=-1, high=1, shape=(2,)),
#             'yaw': spaces.Box(low=-1, high=1, shape=(2, ))
#         })
        
#         self.action_space = spaces.Dict({
#             'position': spaces.Box(low=-half, high=half, shape=(2,), dtype=np.float32),
#             'orientation': spaces.Box(low=-pi, high=np.pi, dtype=np.float32)
#         })

        self.seed()
        
        half = self.slam_map_size*self.slam_resolution/2
        self.movable_range = half/5
        
        self.exist_initial_room = False
        
        self.action_space = spaces.Box(low=np.full([3], -1.0), high=np.full([3], 1.0))
        self.action_range = np.array([self.movable_range, np.pi/2, np.pi/2])
        
        self.map_trueth = []
        self.start_frame = [0,0,0] # initial pose [x,y,yaw] in world frame when started episode 
        self.agent_pose = [0,0,0] # now pose [x,y,yaw] in world frame
        self.target_num = 0
        self.target_pose = [] # target poses [[x,y,yaw],] in world frame
        self.target_found = [] # flag that target have been found for each
        
        self.goal_pose = [] # pose [x,y,yaw] that agent was going to go to
        self.goal_threshold = [0.5, 0.5, 0.5]
        self.is_reached_goal = False
        self.is_done_action = False
        self.episode_start_time = 0
        
        self.time_taken_action = 0 # time between send action and receive response of result from Robot Server
        self.total_time = 0
        
        # Connect to Robot Server
        if rs_address:
            self.client = rs_client.Client(rs_address)
        else:
            print("WARNING: No IP and Port passed. Simulation will not be started")
            print("WARNING: Use this only to get environment shape")
            
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self, 
              new_room: bool=False,
              new_agent_pose: bool=True, 
              obstacle_count: int=20, 
              obstacle_size: float=0.4, 
              target_size: float=0.2, 
              room_length_max: float=10.0, 
              room_mass_min: float=55.0, 
              room_mass_max: float=60.0, 
              wall_height: float=0.8,
              room_wall_thickness: float=0.05,
              target_poses:List[List[float]]=None):
        
        """Environment reset
        
        Args:
            new_room (bool): is generate new room when initialize Environment
            new_agent_pose (bool): is change pose in the room when initialize Environment
        """
        
        self.elapsed_steps = 0
        self.prev_base_reward = None
        
        # Initialize environment state
        self.state = {}
        rs_state = np.zeros(self._get_robot_server_state_len())
        
        ignore_start = 1
        map_state_len = (self.map_size**2)*2
        ignore_len = map_state_len + 6
        ignore_index = ignore_start + ignore_len
        
        if not self.exist_initial_room:
            new_room = True
            self.exist_initial_room = True
        
        rs_state[0] = self.map_size
        rs_state[ignore_index] = new_room
        rs_state[ignore_index+1] = new_agent_pose
        rs_state[ignore_index+2] = obstacle_count
        rs_state[ignore_index+3] = obstacle_size
        rs_state[ignore_index+4] = target_size
        rs_state[ignore_index+5] = room_length_max
        rs_state[ignore_index+6] = room_mass_min
        rs_state[ignore_index+7] = room_mass_max
        rs_state[ignore_index+8] = wall_height
        rs_state[ignore_index+9] = room_wall_thickness
        
        state_msg = robot_server_pb2.State(state=rs_state)
        if not self.client.set_state_msg(state_msg):
            raise RobotServerError("set_state")
            
        # Get Robot Server state
        rs_state = copy.deepcopy(np.array(self.client.get_state_msg().state))

        # in World frame
        self.start_frame = rs_state[1+map_state_len : 1+map_state_len+3]
        
        assert len(rs_state[ignore_index+10:]) % 3 == 0
        self.target_num = len(rs_state[ignore_index+10:])//3
        self.target_found = np.full([self.target_num], False)
            
        self.agent_pose = np.array(self.start_frame) # [x,y,yaw] pose in world frame
        self.target_pose = np.reshape(rs_state[ignore_index+10:], (self.target_num, 3))
        
        self.agent_twist = rs_state[2+map_state_len : 2+map_state_len+2]
        self.map_trueth = rs_state[1+self.map_size**2 : 1+map_state_len]
        
        self.state = self._robot_server_state_to_env_state(rs_state)
        
        self.is_done_action = False
        self.is_reached_goal = False
        self.episode_start_time = 0
        self.total_time = 0

        # Check if the environment state is contained in the observation space
        if not self.observation_space.contains(self.state):
            raise InvalidStateError()
        return self.state
    
    def _reward(self, rs_state, action):
        return 0, False, {}
    
    def step(self, action):
        self.elapsed_steps += 1
        if not self.is_done_action:
            self.episode_start_time = time.time()
            self.is_done_action = True
        
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        
        rs_action = copy.deepcopy(action)
        # Scale action
        rs_action = np.multiply(rs_action, self.action_range)
        # Polar to Cartesian
        x, y = polar_to_cartesian_2d(rs_action[0], rs_action[1])
        rs_action = [x, y, rs_action[2]]
        # Transformate coordinates of agent frame to map frame
        map_trans = polar_to_cartesian_2d(*self.state['agent_pose'][:2])
        rs_action = relative_to_origin(
            rs_action[0], rs_action[1], rs_action[2], 
            map_trans[0], map_trans[1], self.state['agent_pose'][2]
        )
        
        # ideal goal pose in world frame
        self.goal_pose = relative_to_origin(rs_action[0],rs_action[1], rs_action[2], *self.start_frame)
            
        start = time.time()
        
        # Send action to Robot Server
        if not self.client.send_action(rs_action):
            raise RobotServerError("send_action")
            
        self.time_taken_action = time.time() - start
        self.total_time += self.time_taken_action
        
        # Get state from Robot Server
        rs_state = self.client.get_state_msg().state
        # Convert the state from Robot Server format to environment format
        self.state = self._robot_server_state_to_env_state(rs_state)
        # Set agent_pose in world frame
        self.agent_pose = self._squeeze_agent_pose(rs_state)
        
        self.is_reached_goal = self._check_goal(self.goal_pose, self.agent_pose)
        
        # Check if the environment state is contained in the observation space
        if not self.observation_space.contains(self.state):
            raise InvalidStateError()
        
        # Assign reward
        reward, done, info = self._reward(rs_state=rs_state, action=action)

        return self.state, reward, done, info
    
    def render(self):
        pass
        
    def _get_env_state_len(self) -> int:
        ## State include occupancy grid data and mir pose [x,y,yaw] in map frame 
        map_data = [0] * self.map_size**2
        r_theta_yaw = [0.0, 0.0, 0.0]

        env_state = map_data + r_theta_yaw
        
        return len(env_state)
    
    def _get_robot_server_state_len(self) -> int:
#         map_size = [0]
#         map_data = [0] * self.map_size**2
#         map_data_trueth = [0] * self.map_size**2
#         agent_pose = [0] * 3
#         agent_twist = [0] * 2
#         is_collision = [0]
#         is_change_room = [0]
#         is_change_pose = [0]
#         room_generator_param = [0] * 8
        
#         rs_state = map_size + map_data + map_data_trueth + agent_pose + agent_twist + is_collision \
#                     + is_change_room + is_change_pose + room_generator_param
        
        return (self.map_size**2)*2 + 17
    
    def _check_goal(self, ideal_pose, actual_pose) -> bool:
        diff = np.array(ideal_pose) - np.array(actual_pose)
        return all(np.abs(diff) < self.goal_threshold)
    
    def _squeeze_agent_pose(self, rs_state):
        map_state_len = (self.map_size**2)*2
        x, y, yaw = rs_state[map_state_len+1 : map_state_len+4]
        return x, y, yaw
    
    def _robot_server_state_to_env_state(self, rs_state):
        pose = self._squeeze_agent_pose(rs_state)
        odom_x, odom_y, yaw = transform_2d(pose[0], pose[1], pose[2], *self.start_frame)
        polar_r, polar_theta = utils.cartesian_to_polar_2d(x_target=odom_x, y_target=odom_y)
        
        # Normalize to +/- pi
        polar_theta = utils.normalize_angle_rad(polar_theta)
        
#         state = np.concatenate([rs_state[1:self.map_size**2], [polar_r, polar_theta, yaw]])
        
        state = {
            'occupancy_grid': np.array(rs_state[1:1+self.map_size**2], dtype=np.float32),
            'agent_pose': np.array([polar_r, polar_theta, yaw])
        }

        return state
    
    def _get_observation_space(self):
        occupancy_grid_space = spaces.Box(low=-1, high=100, shape=(self.map_size**2,), dtype=np.float32)
        
        min_polar_r = 0
        max_polar_r = np.inf
        min_polar_theta = -np.pi
        max_polar_theta = np.pi
        min_yaw = -np.pi
        max_yaw = np.pi
        
        min_pose_obs = np.array([min_polar_r, min_polar_theta, min_yaw])
        max_pose_obs = np.array([max_polar_r, max_polar_theta, max_yaw])
        agent_pose_space = spaces.Box(low=min_pose_obs, high=max_pose_obs, dtype=np.float32)
        
        observation_space = spaces.Dict({
            'occupancy_grid': occupancy_grid_space,
            'agent_pose': agent_pose_space,
        })
        
        return observation_space
    
class CubeRoomOnNavigationStack(Mir100NavEnv, Simulation):
    cmd = "roslaunch task_on_nav_robot_server sim_robot_server.launch"
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        Mir100NavEnv.__init__(self, rs_address=self.robot_server_ip, **kwargs)
    
class CubeRoomSearch(Mir100NavEnv, Simulation):
    cmd = "roslaunch task_on_nav_robot_server sim_robot_server.launch wait_moved:=true"
    
    found_thresh = 0.75
    too_long_time_thresh = 6
    total_time_limit = 120
    
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        Mir100NavEnv.__init__(self, rs_address=self.robot_server_ip, **kwargs)
        
    def _reward(self, rs_state, action):
        reward = -0.05
        done = False
        info = {}
        
        # check if there is target found
        diff_targets = np.full([self.target_num,2], self.agent_pose[:2]) - self.target_pose[:,:2]
        dist_targets = np.linalg.norm(diff_targets, axis=1)
        are_found = np.logical_and(dist_targets<=self.found_thresh, np.logical_not(self.target_found))
        idx_found = np.arange(self.target_num)[are_found]
        
        if len(idx_found) > 0:
            self.target_found[idx_found[0]] = True
            reward += 10.0
            
        if not self.is_reached_goal:
            reward -= 0.5
            
        if self.time_taken_action > self.too_long_time_thresh:
            reward -= 0.5
            
        done1 = np.sum(self.target_found) == self.target_num
        done2 = self.total_time > self.total_time_limit
            
        if done1:
            info['final_status'] = 'success'
        elif done2:
            info['final_status'] = 'max_steps_exceeded'
            
        done = done1 or done2
            
        return reward, done, info
        
class CubeRoomSearchLikeContinuously(Mir100NavEnv, Simulation):
    wait_for_current_action = 5
    
    cmd = f"roslaunch task_on_nav_robot_server sim_robot_server.launch wait_moved:=false sleep_time:={wait_for_current_action}"
    
    found_thresh = 0.75
    
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        Mir100NavEnv.__init__(self, rs_address=self.robot_server_ip, **kwargs)
        
    def _reward(self, rs_state, action):
        reward = -0.05
        done = False
        info = {}
        
        # check if there is target found
        diff_targets = np.full([self.target_num,2], self.agent_pose[:2]) - self.target_pose[:,:2]
        dist_targets = np.linalg.norm(diff_targets, axis=1)
        are_found = np.logical_and(dist_targets<=self.found_thresh, np.logical_not(self.target_found))
        idx_found = np.arange(self.target_num)[are_found]
        
        if len(idx_found) > 0:
            self.target_found[idx_found[0]] = True
            reward += 50.0
            
        if self.is_reached_goal:
            reward += 0.05
            
        if np.sum(self.target_found) == self.target_num:
            done = True
            info['final_status'] = 'success'
            
        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info['final_status'] = 'max_steps_exceeded'

        return reward, done, info
        