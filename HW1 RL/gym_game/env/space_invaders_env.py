import gym
from gym import spaces
import numpy as np

class SpaceInvadersEnv(gym.Env):

    def __init__(self,):
        # gym attribute
        self.action_space = spaces.Discrete(3)  # left, right,, shoot
        self.observation_space = spaces.Box(low=0, high=1, shape=[7, 7])


        self.cols = self.observation_space.shape[0]
        self.rows = self.observation_space.shape[1]
        self.step_cost = 1
        self.goal = 0

        # create rewards table
        self.rewards = np.zeros((self.rows, self.cols))
        self.rewards[self.rows - 1] = - self. step_cost  # step cost of moves
        self.shootReward = 0

        # creation of invaders
        self.initial_number_invaders = 7
        self.createinvaders()

        for r, c in self.invaders:
            self.rewards[r, c] = + 0.5

        # initialization of the matrix of states and definition of the current state
        self.states = np.zeros((self.rows, self.cols))
        self.init_state = (self.rows - 1, 3, self.invaders)
        self.current_state = self.init_state

        # handle of the agent's actions
        self.action_semantics = ['left', 'right', 'shoot']
        self.shoot_property = {'row': 4}
        self.actions = np.array(
            [[0, -1], [0, 1], self.shoot_property])  # shoot dictionary for the activation of the shoot func.

        # define the transition model P(s'|s,a)
        self._transition_probabilities = np.array([0.1, 0.8, 0.1])

    def createinvaders(self, full= False):
        invs = []
        if full:
            for a in range(5):
                for b in range(7):
                    invs.append((a, b))
        else:
            draw_number = 0
            while draw_number < self.initial_number_invaders:
                m = np.random.randint(0,5)
                n= np.random.randint(0,7)
                if invs.__contains__((m,n)):
                    continue
                else:
                    draw_number += 1
                    invs.append((m,n))
        self.invaders = invs


    def reset(self):
        self.current_state = self.init_state

    def termination(self):
        r, l, c = self.current_state
        return len(c) == self.goal  # killed all the invaders

    def shoot(self, row):
        x_pos = self.current_state[1]
        if row == -1:  # end of the grid
            return self.shootReward
        else:
            if self.rewards[row, x_pos] == 0.5:
                r, c, invaders = self.current_state
                invaders.remove((row, x_pos))
                self.shootReward = self.rewards[row, x_pos]  # the ship has hit an invader
                self.rewards[row, x_pos] = 0.0
                return self.shootReward
            else:
                self.shoot((row - 1))

    def reward(self) -> float:
        r, c, l = self.current_state
        if self.shootReward != 0:  # shoot reward
            return self.shootReward
        else:  # move reward
            return self.rewards[r, c]

    def transition(self, state, a):

        action = self.actions[a]
        if a == 2:   # the shoot action has been chosen
            self.shoot(action['row'])
            return self.current_state
        else:
            new_c = state[1] + action[1]
            # check if new position is out of bounds
            return (state[0], new_c, state[2]) if (new_c >= 0) and (new_c <= self.cols-1) else state

    def step(self, a_idx):
        self.shootReward = 0   # in every step the actual reward of the shoot action is resetted

        chosen_action = a_idx + np.random.choice([1, 0, -1], p=self._transition_probabilities)   #probability of doing that action
        n_actions = 3
        effective_action = chosen_action + n_actions if chosen_action < 0 else chosen_action % n_actions  # module the index of the action

        prev_state = self.current_state
        self.current_state = self.transition(self.current_state, effective_action)

        reward = self.reward()
        done = self.termination()
        observation = self.current_state
        info = {"effective_action": effective_action,"sampled_action": a_idx, "prev_state": prev_state}

        return observation, reward, done, info


    def render(self, mode='human'):
        grid = np.array(self.states, dtype=str)
        r, c, invaders = self.current_state
        grid[r, c] = ' X '

        for x, y in invaders:
            grid[x, y] = ' I '

        print(grid)



