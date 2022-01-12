


class ConstantReward:
    def __init__(self, reward_size):
        self.reward_size = reward_size

    def __call__(self, intervention_mag, obs):
        if intervention_mag == 0:
            return 0
        return -self.reward_size + obs['reward']


class ConstantContinuousReward:
    def __init__(self, reward_size):
        self.reward_size = reward_size

    def __call__(self, intervention_mag, obs):
        if intervention_mag == 0:
            return 0
        return -self.reward_size 

class ZeroReward:
    def __call__(self, intervention_mag, obs):
        return 0

class MagnitudeReward:
    def __init__(self, reward_scale):
        self.reward_scale = reward_scale

    def __call__(self, intervention_mag, obs):
        return - self.reward_scale * abs(intervention_mag) + obs['reward']

