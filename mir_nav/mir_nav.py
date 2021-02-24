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

from .utils import transform_2d, cartesian_to_polar_2d, polar_to_cartesian_2d

from typing import List

class Mir100NavEnv(gym.Env):
    real_robot = False
    map_size = 256
    resolution = 0.05
    
    def __init__(self, rs_address=None, max_episode_steps=500, **kwargs):
        self.max_episde_steps = max_episode_steps
        self.elapsed_steps = 0
        
        self.observation_space = self._get_observation_space()
        
#         self.action_space = spaces.Dict({
#             'polar_r': spaces.Box(low=0, high=1, shape=(2,)),
#             'polar_theta': spaces.Box(low=-1, high=1, shape=(2,)),
#             'yaw': spaces.Box(low=-1, high=1, shape=(2, ))
#         })
        
#         self.action_space = spaces.Dict({
#             'position': spaces.Box(low=-half, high=half, shape=(2,), dtype=np.float32),
#             'orientation': spaces.Box(low=0, high=np.pi*2, dtype=np.float32)
#         })

        self.seed()
        self.distance_threshold = 0.2
        self.min_target_dist = 1.0
        
        half = map_size*resolution/2
        self.movable_range = half/2
        
        self.action_space = spaces.Box(low=np.array([0,-1,-1]), high=np.array([1,1,1]))
        self.action_range = np.array([self.movable_range, np.pi, np.pi])
        
        self.map_trueth = []
        self.start_frame = [0,0,0] # initial pose [x,y,yaw] in world frame when started episode 
        self.agent_pose = [0,0,0] # now pose [x,y,yaw] in map frame
        self.target_num = 0
        self.target_pose = [] # target poses [[x,y,yaw],] in world frame
        
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
              new_room: bool,
              new_agent_pose: bool, 
              obstacle_count: int=10, 
              obstacle_size: float=0.4, 
              target_size: float=0.2, 
              room_length_max: float=8.0, 
              room_mass_min: float=36.0, 
              room_mass_max: float=40.0, 
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
        
        self.target_num = len(rs_state[ignore_index+9:])//3
        if len(rs_state[ignore_index+9:]) % 3 != 0:
            raise Exception("wrong length of targets in robot server state")
            
        self.agent_pose = np.array([0, 0, 0]) # [x,y,yaw] pose in map frame
        self.target_pose = np.reshape(rs_state[ignore_index+9:], [self.target_num, 3])
        self.agent_twist = rs_state[2+map_state_len : 2+map_state_len+2]
        self.map_trueth = rs_state[1+self.map_size**2 : 1+map_state_len]
        
        self.state = self._robot_server_state_to_env_state(rs_state)

        # Check if the environment state is contained in the observation space
        if not self.observation_space.contains(self.state):
            raise InvalidStateError()
            
        return self.state
    
    def _reward(self, rs_state, action):
        return 0, False, {}
    
    def step(self, action):
        self.elapsed_steps += 1
        
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        
        rs_action = copy.deepcopy(action)
        # Scale action
        rs_action = np.multiply(rs_action, self.action_range)
        # Polar to Cartesian
        x, y = polar_to_cartesian_2d(rs_action[0], rs_action[1])
        rs_action = [x, y, rs_action[2]]
        # Transformate coordinates of agent frame to map frame
        rs_action = transform_2d(rs_action[0], rs_action[1], rs_action[2], *self.agent_pose)
        
        # Send action to Robot Server
        if not self.client.send_action(rs_action.tolist()):
            raise RobotServerError("send_action")
        
        # Get state from Robot Server
        rs_state = self.client.get_state_msg().state
        # Convert the state from Robot Server format to environment format
        self.state = self._robot_server_state_to_env_state(rs_state)
        # Set agent_pose in map frame
        self.agent_pose = polar_to_cartesian_2d(
            self.state['agent_pose'][0],
            self.state['agent_pose'][1],
            self.state['agent_pose'][2],
            *self.start_frame
        )
        
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
    
    def _robot_server_state_to_env_state(self, rs_state):
        map_state_len = (self.map_size**2)*2
        pose = rs_state[map_state_len+1 : map_state_len+4]
        odom_x, odom_y, yaw = transform_2d(pose[0], pose[1], pose[2], *self.start_frame)
        polar_r, polar_theta = utils.cartesian_to_polar_2d(x_target=odom_x, y_target=odom_y)
        
        # Normalize to +/- pi
        polar_theta = utils.normalize_angle_rad(polar_theta)
        
#         state = np.concatenate([rs_state[1:self.map_size**2], [polar_r, polar_theta, yaw]])
        
        state = {
            'occupancy_grid': np.array(rs_state[1:self.map_size**2]),
            'agent_pose': np.array([polar_r, polar_theta, yaw])
        }

        return state
    
    def _get_observation_space(self):
        occupancy_grid_space = spaces.Box(low=0, high=256, shape=(self.map_size**2,), dtype=np.int16)
        
        min_polar_r = 0
        max_polar_r = np.inf
        min_polar_theta = -np.pi
        max_polar_theta = np.pi
        min_yaw = 0
        max_yaw = np.pi*2
        
        min_pose_obs = np.array([min_polar_r, min_polar_theta, min_yaw])
        max_pose_obs = np.array([max_polar_r, max_polar_theta, max_yaw])
        agent_pose_space = spaces.Box(low=min_pose_obs, high=max_pose_obs)
        
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