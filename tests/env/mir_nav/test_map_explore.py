import gym
import robo_gym
import pytest

@pytest.fixture(scope='module')
def env(request):
    env = gym.make('MapExploreInCubeRoomObsMapOnly-v0', ip='robot-server')
    yield env
    env.kill_sim()

@pytest.mark.commit 
def test_initialization(env):
    env.reset(new_room=True, new_agent_pose=True)
    done = False
    for _ in range(10):
        if not done:
            action = env.action_space.sample()
            observation, _, done, _ = env.step(action)

    assert env.observation_space.contains(observation)

