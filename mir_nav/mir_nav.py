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

from collections import OrderedDict, deque

def create_slice_dict(len_dict):
    assert len([x for x in len_dict.values() if x <= 0]) <= 1
    
    start = 0
    slice_dict = {}
    for key in len_dict:
        end = start+len_dict[key] if len_dict[key] > 0 else None
        slice_dict[key] = slice(start, end)
        start += len_dict[key]
        
    return slice_dict

class Mir100NavEnv(gym.Env):
    real_robot = False
    slam_map_size = 512
    slam_resolution = 0.05
    map_size = 128
    resolution = slam_resolution * (slam_map_size / map_size)

    rs_state_len_dict = OrderedDict(
        map_size=1,
        map_data=map_size**2,
        trueth_map_data=map_size**2,
        agent_pose=3,
        agent_twist=2,
        is_agent_collisioned=1,
        new_room_flag=1,
        new_agent_pose_flag=1,
        room_generator_param=8,
        target_poses=-1
    )

    rs_state_slice_dict = create_slice_dict(rs_state_len_dict)
    
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
        self.action_range = np.array([self.movable_range, np.pi/2, np.pi])
        
        self.map_trueth = []
        self.start_frame = [0,0,0] # initial pose [x,y,yaw] in world frame when started episode 
        self.agent_pose = [0,0,0] # now pose [x,y,yaw] in world frame
        self.target_num = 0
        self.target_pose = [] # target poses [[x,y,yaw],] in world frame
        
        self.goal_pose = [] # pose [x,y,yaw] that agent was going to go to
        self.goal_threshold = [0.5, 0.5, 0.5]
        self.is_reached_goal = False
        self.is_done_action = False
        self.episode_start_time = 0
        
        self.time_taken_action = 0 # time between send action and receive response of result from Robot Server
        self.total_time = 0
        
        self.move_distance = 0
        
        # Connect to Robot Server
        if rs_address:
            self.client = rs_client.Client(rs_address)
        else:
            print("WARNING: No IP and Port passed. Simulation will not be started")
            print("WARNING: Use this only to get environment shape")
            
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def get_from_rs_state(self, rs_state, key: str):
        assert key in self.rs_state_slice_dict.keys()

        return rs_state[self.rs_state_slice_dict[key]]

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
            obstacle_count (int): count of cube putting in the room as obstacle
            obstacle_size (float): size of cube as obstacle
            target_size (float): size of cube as target
            room_length_max (float): maximum value of room's length
            room_mass_min (float): minimum value of room's area
            room_mass_max (float): maximum value of room's area
            wall_height (float): height of room's wall
            room_wall_thickness (float): thickness of room's wall
        """
        
        print(f"Resetting env... [room: {new_room}, pose: {new_agent_pose}]")
        
        self.elapsed_steps = 0
        self.prev_base_reward = None
        
        # Initialize environment state
        self.state = {}
        rs_state = np.zeros(self._get_robot_server_state_len())
        
        if not self.exist_initial_room:
            new_room = True
            self.exist_initial_room = True
        
        rs_state[self.rs_state_slice_dict['map_size']] = [self.map_size]
        rs_state[self.rs_state_slice_dict['new_room_flag']] = [new_room]
        rs_state[self.rs_state_slice_dict['new_agent_pose_flag']] = [new_agent_pose]
        rs_state[self.rs_state_slice_dict['room_generator_param']] = [
            obstacle_count,
            obstacle_size,
            target_size,
            room_length_max,
            room_mass_min,
            room_mass_max,
            wall_height,
            room_wall_thickness
        ]
        
        state_msg = robot_server_pb2.State(state=rs_state)
        if not self.client.set_state_msg(state_msg):
            raise RobotServerError("set_state")
            
        # Get Robot Server state
        rs_state = copy.deepcopy(np.array(self.client.get_state_msg().state))

        # in World frame
        self.start_frame = self.get_from_rs_state(rs_state, 'agent_pose')
        
        last_states = self.get_from_rs_state(rs_state, 'target_poses')
        assert len(last_states) % 3 == 0
        self.target_num = len(last_states)//3
        
            
        self.agent_pose = np.array(self.start_frame) # [x,y,yaw] pose in world frame
        self.target_pose = np.reshape(last_states, (self.target_num, 3))
        
        self.agent_twist = self.get_from_rs_state(rs_state, 'agent_twist')
        self.map_trueth = self.get_from_rs_state(rs_state, 'trueth_map_data')
        
        self.state = self._robot_server_state_to_env_state(rs_state)
        
        self.is_done_action = False
        self.is_reached_goal = False
        self.episode_start_time = 0
        self.total_time = 0
        self.move_distance = 0

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
            
        start_pose = self.agent_pose
        
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        
        rs_action = copy.deepcopy(action)
        # Scale action
        rs_action = np.multiply(rs_action, self.action_range)
        # Polar to Cartesian
        x, y = polar_to_cartesian_2d(rs_action[0], rs_action[1])
        rs_action = [x, y, rs_action[2]]
        # Transform coordinates of agent frame to map frame
        odom_x, odom_y, yaw = transform_2d(self.agent_pose[0], self.agent_pose[1], self.agent_pose[2], *self.start_frame)
        # calc relative pose
        rs_action = relative_to_origin(
            rs_action[0], rs_action[1], rs_action[2], 
            odom_x, odom_y, yaw
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
        self.agent_pose = self.get_from_rs_state(rs_state, 'agent_pose')
        
        self.move_distance = np.linalg.norm(np.array(self.agent_pose[:2]) - np.array(start_pose[:2]))
        
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
        ## State include occupancy grid data and mir pose [r,theta,yaw] in map frame 
        map_data = [0] * self.map_size**2
        r_theta_yaw = [0.0, 0.0, 0.0, 0.0, 0.0] # [r, theta_sin, theta_cos, yaw_sin, yaw_cos]

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
        
        # return (self.map_size**2)*2 + 17

        return sum([x for x in self.rs_state_len_dict.values() if x > 0])
    
    def _check_goal(self, ideal_pose, actual_pose) -> bool:
        diff = np.array(ideal_pose) - np.array(actual_pose)
        return all(np.abs(diff) < self.goal_threshold)
    
    def _robot_server_state_to_env_state(self, rs_state):
        pose = self.get_from_rs_state(rs_state, 'agent_pose')
        odom_x, odom_y, yaw = transform_2d(pose[0], pose[1], pose[2], *self.start_frame)
        polar_r, polar_theta = utils.cartesian_to_polar_2d(x_target=odom_x, y_target=odom_y)
        
        # Normalize to +/- pi
        polar_theta = utils.normalize_angle_rad(polar_theta)
        
        polar_theta_sin = np.sin(polar_theta)
        polar_theta_cos = np.cos(polar_theta)
        yaw_sin = np.sin(yaw)
        yaw_cos = np.cos(yaw)
        
        state = {
            'occupancy_grid': np.array(self.get_from_rs_state(rs_state, 'map_data'), dtype=np.float32),
            'agent_pose': np.array([polar_r, polar_theta_sin, polar_theta_cos, yaw_sin, yaw_cos], dtype=np.float32)
        }

        return state
    
    def _get_observation_space(self):
        occupancy_grid_space = spaces.Box(low=-1, high=100, shape=(self.rs_state_len_dict['map_data'],), dtype=np.float32)
        
        min_polar_r = 0
        min_polar_theta_sin = -1
        min_polar_theta_cos = -1
        min_yaw_sin = -1
        min_yaw_cos = -1

        max_polar_r = np.inf
        max_polar_theta_sin = 1
        max_polar_theta_cos = 1
        max_yaw_sin = 1
        max_yaw_cos = 1
        
        min_pose_obs = np.array([min_polar_r, min_polar_theta_sin, min_polar_theta_cos, min_yaw_sin, min_yaw_cos])
        max_pose_obs = np.array([max_polar_r, max_polar_theta_sin, max_polar_theta_cos, max_yaw_sin, max_yaw_cos])
        agent_pose_space = spaces.Box(low=min_pose_obs, high=max_pose_obs, dtype=np.float32)
        
        observation_space = spaces.Dict({
            'occupancy_grid': occupancy_grid_space,
            'agent_pose': agent_pose_space,
        })
        
        return observation_space

class CubeRoomWithTargetFind(Mir100NavEnv):
    
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

class CubeRoomWithMapDifferenceCalculate(Mir100NavEnv):
    
    def __init__(self, *args, **kwargs):
        Mir100NavEnv.__init__(self, *args, **kwargs)

        self.map_queue = deque(maxlen=2)

    def reset(self, *args, **kwargs):
        state = super().reset(*args, **kwargs)
        self.map_queue = deque(maxlen=2)
        return state

    def step(self, action):
        self.map_queue.append(self.state['occupancy_grid'])
        state, reward, done, info = super().step(action)
        return state, reward, done, info

    def calculate_both_maps_diff(self) -> int:
        '''
        Return how much decreased the unknown area as pixel num.
        This function is supposed to be called in the reward function.
        '''
        return np.sum(self.map_queue[-1] < 0) - np.sum(self.state['occupancy_grid'] < 0)

    
class CubeRoomOnNavigationStack(Mir100NavEnv, Simulation):
    cmd = "roslaunch task_on_nav_robot_server sim_robot_server.launch"
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        Mir100NavEnv.__init__(self, rs_address=self.robot_server_ip, **kwargs)
    
class CubeRoomSearch(CubeRoomWithTargetFind, Simulation):
    cmd = "roslaunch task_on_nav_robot_server sim_robot_server.launch wait_moved:=true"
    
    found_thresh = 0.75
    too_long_time_thresh = 6
    total_time_limit = 120
    
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        CubeRoomWithTargetFind.__init__(self, rs_address=self.robot_server_ip, **kwargs)
        
    def _reward(self, rs_state, action):
        reward = -0.05
        done = False
        info = {}
        
        is_found, _ = self.check_found_new_one(threshold=self.found_thresh)
        
        if is_found:
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
        
class CubeRoomSearchLikeContinuously(CubeRoomWithTargetFind, Simulation):
    wait_for_current_action = 5
    found_thresh = 0.75
    move_distance_thresh = 0.7
    
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, gazebo_gui=False, **kwargs):
        opt_wait_moved = 'wait_moved:=false'
        opt_sleep_time = f'sleep_time:={self.wait_for_current_action}'
        opt_gazebo_gui = f'gazebo_gui:={"true" if gazebo_gui else "false"}'
        self.cmd = f"roslaunch task_on_nav_robot_server sim_robot_server.launch {opt_wait_moved} {opt_sleep_time} {opt_gazebo_gui}"

        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        CubeRoomWithTargetFind.__init__(self, rs_address=self.robot_server_ip, **kwargs)
        
    def _reward(self, rs_state, action):
        reward = -0.05
        done = False
        info = {}
        
        is_found, _ = self.check_found_new_one(threshold=self.found_thresh)
        
        if is_found:
            reward += 50.0

        if self.move_distance > self.move_distance_thresh:
            reward += 0.05
            
        if np.sum(self.target_found) == self.target_num:
            done = True
            info['final_status'] = 'success'
            
        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info['final_status'] = 'max_steps_exceeded'

        return reward, done, info
        

class CubeRoomMapExplorationLikeContinuously(CubeRoomWithMapDifferenceCalculate, Simulation):
    wait_for_current_action = 5
    
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, gazebo_gui=False, **kwargs):
        opt_wait_moved = 'wait_moved:=false'
        opt_sleep_time = f'sleep_time:={self.wait_for_current_action}'
        opt_gazebo_gui = f'gazebo_gui:={"true" if gazebo_gui else "false"}'
        self.cmd = f"roslaunch task_on_nav_robot_server sim_robot_server.launch {opt_wait_moved} {opt_sleep_time} {opt_gazebo_gui}"

        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        CubeRoomWithTargetFind.__init__(self, rs_address=self.robot_server_ip, **kwargs)
        
    def _reward(self, rs_state, action):
        reward = -0.05
        done = False
        info = {}
        
        explored_pixel_num = self.calculate_both_maps_diff()

        reward += explored_pixel_num

        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info['final_status'] = 'max_steps_exceeded'

        return reward, done, info