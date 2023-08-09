import pickle


class TrajectoryInstance:
    """ Data class instance """

    def __init__(self, trajectory, actions, rewards, seed=None):
        self.trajectory = trajectory
        self.actions = actions
        self.rewards = rewards
        self.seed = seed  # seed used to create instance

    def to_file(self, filepath):
        """ Write this instance to a pickle file. """
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def from_file(filepath):
        """ Load trajectory learning instance from a pickled file. """
        with open(filepath, "rb") as f:
            inst = pickle.load(f)
            return inst

    @property
    def total_reward(self):
        return sum(self.rewards)


class CompleteTrajectoryInstance(TrajectoryInstance):
    """ A complete trajectory where everything is observed. """

    def __init__(self, trajectory, actions, rewards, seed):
        super().__init__(trajectory, actions, rewards, seed)
        assert len(self.trajectory) > 1
        assert len(self.actions) == len(self.trajectory) - 1
        assert len(self.actions) == len(self.rewards)

    def __str__(self):
        result = f"seed:{self.seed}"
        for index in range(len(self.trajectory)-1):
            result += (f"state: {self.trajectory[index]}\n"
                       f"action: {self.actions[index]}\n"
                       f"gave reward {self.rewards[index]}\n")
        result += f"state:{self.trajectory[-1]}"
        return result

    @staticmethod
    def from_trajectory(inst: TrajectoryInstance):
        return CompleteTrajectoryInstance(inst.trajectory, inst.actions, inst.rewards, inst.seed)


class LearnTrajectoryInstance(TrajectoryInstance):
    """
    A Trajectory instance used for learning, i.e., where the trajectory consists of
        only the initial state.
    """
    def __init__(self, init_state, actions, rewards, seed):
        super().__init__([init_state], actions, rewards, seed)
        assert len(self.actions) == len(self.rewards)

    @property
    def init_state(self):
        return self.trajectory[0]

    def __str__(self):
        result = f"seed:{self.seed}\n{self.trajectory}\n"
        for index, action in enumerate(self.actions):
            result += f"reward: {self.rewards[index]} after {action}\n"
        return result

    @staticmethod
    def from_complete_trajectory(inst: CompleteTrajectoryInstance):
        return LearnTrajectoryInstance(inst.trajectory[0], inst.actions, inst.rewards, inst.seed)
