from collections import OrderedDict
import numpy as np
from gym.spaces import Dict, Box
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.core.serializable import Serializable
from multiworld.envs.env_util import get_stat_in_paths, create_stats_ordered_dict, get_asset_full_path
from multiworld.envs.mujoco.mujoco_env import MujocoEnv
from pyquaternion import Quaternion

try:
    import torch
except ImportError:
    pass

class AntEnv(MujocoEnv, MultitaskEnv, Serializable):
    def __init__(self, action_scale=1, frame_skip=5, reward_type='vel_distance', indicator_threshold=0.1, fixed_goal=5, fix_goal=False, max_speed=6, 
                 use_vel_in_goal=False, use_vel_in_state=True, history_len = 3,
                ):
        self.quick_init(locals())
        MultitaskEnv.__init__(self)
        self.action_scale = action_scale
        MujocoEnv.__init__(self, self.model_name, frame_skip=frame_skip)
        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = Box(low=low, high=high)
        self.reward_type = reward_type
        self.indicator_threshold=indicator_threshold
        self.fixed_goal = fixed_goal
        self.fix_goal = fix_goal
        self._state_goal = None
        self.use_vel_in_goal = use_vel_in_goal
        self.use_vel_in_state = use_vel_in_state
        self.history_len = history_len
        self.history  = []
        obs_size = self._get_obs()['observation'].shape[0]

        # Angles are in degrees
        high = np.array([ 1, 1, 2,1, 180, 180, 360, 30,70, 30,-30, 30,-30, 30,70])
        low =  np.array([-1,-1,0.5,-1, -180,-180,-360,-30,30,-30,-70,-30,-70,-30,30])
        
        high_vel = np.zeros(high.shape[0])
        high_tot = np.concatenate((high, high_vel))
        low_tot = np.concatenate((low, -1*high_vel))
        if self.use_vel_in_goal:
            self.goal_space = Box(low_tot, high_tot)
        else:
            self.goal_space = Box(low, high)# Box(np.array(-1*max_speed), np.array(max_speed))
        
        high = np.inf * np.ones(obs_size)
        low = -high
        self.obs_space = Box(low, high)
        state_size = self._get_obs()['state_observation'].shape[0]
        high = np.inf * np.ones(state_size)
        low = -high
        state_space = Box(low, high)
        self.achieved_goal_space = self.goal_space#Box(self.obs_space.low[8], self.obs_space.high[8])
        self.observation_space = Dict([
            ('observation', self.obs_space),
            ('desired_goal', self.goal_space),
            ('achieved_goal', self.achieved_goal_space),
            ('state_observation', state_space),
            ('state_desired_goal', self.goal_space),
            ('state_achieved_goal', self.achieved_goal_space),
        ])
        self.reset()
        self.imsize = 500

    @property
    def model_name(self):
        return get_asset_full_path('classic_mujoco/ant.xml')

    def step(self, action):
        action = action * self.action_scale
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs(add_to_history=True)
        info = self._get_info()
        reward = self.compute_reward(action, obs)
        done = False
        return obs, reward, done, info

    def _get_env_obs(self):
        if self.use_vel_in_state:
            obs = np.concatenate([self.sim.data.qpos.flat, self.sim.data.qvel.flat])
        else:
            obs =  np.concatenate([self.sim.data.qpos.flat])
            #print(self.sim.data.qpos.flat[3:6], np.linalg.norm(self.sim.data.qpos.flat[3:6]))
        return obs

    def _get_obs(self, add_to_history=False):
        state_obs = self._get_env_obs()
        hist_obs = self._get_obs_with_history(state_obs, add_to_history=add_to_history)
        achieved_goal = state_obs
        if self.use_vel_in_state and not self.use_vel_in_goal:
            achieved_goal = state_obs[:int(state_obs.shape[0]/2)]
        return dict(
            observation=hist_obs,
            desired_goal=self._state_goal,
            achieved_goal=achieved_goal,
            state_observation=state_obs,
            state_desired_goal=self._state_goal,
            state_achieved_goal=achieved_goal,
        )
    
    def _get_obs_with_history(self, state_obs, add_to_history=False):
        obs = state_obs
        hist_obs = [obs]
        if len(hist_obs) < len(self.history) and len(hist_obs) < self.history_len:
            ind = max(-self.history_len+1, -1*len(self.history)+1 )
            hist_obs= self.history[ind:] + hist_obs
        while len(hist_obs) < self.history_len:
            hist_obs.append(obs)
        assert(len(hist_obs) == self.history_len)
        self.history.append(obs)
        return np.concatenate(hist_obs)

    def _get_info(self, ):
        state_obs = self._get_env_obs()
        xpos = state_obs
        if self.use_vel_in_state and not self.use_vel_in_goal:
            xpos = state_obs[:int(state_obs.shape[0]/2)]
        desired_xpos = self._state_goal
        xpos_error = np.linalg.norm(xpos - desired_xpos)
        info = dict()
        info['state_distance'] = xpos_error
        info['state_difference'] =np.abs(xpos - desired_xpos)
        info['state_success'] = (xpos_error < self.indicator_threshold).astype(float)
        info['per_joint_success'] =np.mean((np.abs(xpos - desired_xpos) < self.indicator_threshold).astype(float))
        return info
    
    def weight_difference(self,difference):
        if len(difference.shape) == 1:
            difference[6:15] = difference[6:15]/180.
        else:
            difference[:,6:15] = difference[:,6:15]/180.
        return difference

    def compute_rewards(self, actions, obs):
        achieved_goals = obs['achieved_goal']
        desired_goals = obs['desired_goal']
        difference = achieved_goals - desired_goals
        difference = self.weight_difference(difference)
        if isinstance(achieved_goals,  np.ndarray):
            distances = np.linalg.norm(difference, axis=1)
        else:
            distances = torch.norm(difference, dim=1)
        if self.reward_type == 'vel_distance':
            r = -distances
        elif self.reward_type == 'vel_success':
            r = -(distances > self.indicator_threshold).astype(float)
        else:
            raise NotImplementedError("Invalid/no reward type.")
        return r

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def reset(self):
        self.reset_model()
        self.history = []
        goal = self.sample_goal()
        self._state_goal = goal['state_desired_goal']
        obs = self._get_obs(add_to_history=True)
        return obs

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [
            'state_distance',
            'state_success',
            'state_difference',
            'per_joint_success',
        ]:
            stat_name = stat_name
            stat = get_stat_in_paths(paths, 'env_infos', stat_name)
            statistics.update(create_stats_ordered_dict(
                '%s%s' % (prefix, stat_name),
                stat,
                always_show_all_stats=True,
            ))
            statistics.update(create_stats_ordered_dict(
                'Final %s%s' % (prefix, stat_name),
                [s[-1] for s in stat],
                always_show_all_stats=True,
            ))
        return statistics

    """
    Multitask functions
    """

    @property
    def goal_dim(self) -> int:
        return 1

    def get_goal(self):
        return {
            'desired_goal': self._state_goal,
            'state_desired_goal': self._state_goal,
        }

    def sample_goals(self, batch_size):
        if self.fix_goal:
            goals = np.repeat(
                self.fixed_goal.copy()[None],
                batch_size,
                0
            )
        else:
            goals = np.random.uniform(
                self.goal_space.low,
                self.goal_space.high,
                size=(batch_size, self.goal_space.low.size),
            )
            # Ant has a free joint in indices 0:6, and indices 3:6 are a quaternion.
            # Instead of sampling those value independently, we sample from a 4D unit
            # Gaussian, divide by first value so that it is 1, and pass the last 3 values
            angles = np.random.normal(loc=0.0, scale=1.0, size=(batch_size, 4))
            angles = angles/np.linalg.norm(angles, axis=1)
            goals[:, 3:6] = angles[:,1:]
#             goals_idx = np.random.choice(self.goals.shape[0], size=(batch_size))
#             goals = self.goals[goals_idx]
        return {
            'desired_goal': goals,
            'state_desired_goal': goals,
        }

    def set_to_goal(self, goal):
        pass

    def get_env_state(self):
        joint_state = self.sim.get_state()
        goal = self._state_goal.copy()
        return joint_state, goal

    def set_env_state(self, state):
        state, goal = state
        self.sim.set_state(state)
        self.sim.forward()
        self._state_goal = goal

if __name__ == "__main__":
    env = Ant()
    env.get_goal()
    env.step(np.array(1))
    env.reset()