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
        self.rewards[self.rows - 1] = - self.step_cost  # step cost of moves
        self.shootReward = 0
        self.hit_board_reward = 0

        # creation of invaders
        self.initial_number_invaders = 7
        self.createinvaders()

        for r, c in self.invaders:
            self.rewards[r, c] = + 2.0

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
        self.transition_probabilities = np.array([0.1, 0.8, 0.1])

    def createinvaders(self, is_full= False, is_presetted = True):
        invs = []
        if is_full:
            for a in range(5):
                for b in range(7):
                    invs.append((a, b))
        elif (is_presetted):
            invs= [(3, 6), (3, 0), (3, 1), (4, 1), (1, 1)]
        else:  # random disposition of all the invaders
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
        self.rewards = np.zeros((self.rows, self.cols))
        self.rewards[self.rows - 1] = - self.step_cost  # step cost of moves
        self.shootReward = 0
        self.hit_board_reward = 0

        # creation of invaders
        self.createinvaders()

        for r, c in self.invaders:
            self.rewards[r, c] = + 2.0

        # initialization of the matrix of states and definition of the current state
        self.states = np.zeros((self.rows, self.cols))
        self.init_state = (self.rows - 1, 3, self.invaders)
        self.current_state = self.init_state

# start setter & getter methods
    def getCurrentState(self):
        return self.current_state

    def setCurrentState(self, state):
        self.current_state = state

    def getRewards(self):
        return self.rewards

    def setRewards(self, rewards):
        self.rewards = rewards
# end setter & getter methods

    def termination(self):
        r, l, c = self.current_state
        return len(c) == self.goal  # killed all the invaders

    # shoot action which remove the first invader in front of the space ship
    def shoot(self, row):
        x_pos = self.current_state[1]
        if row == -1:  # end of the grid
            return self.shootReward
        else:
            if self.rewards[row, x_pos] == 2.0:
                r, c, invaders = self.current_state
                invaders.remove((row, x_pos))
                self.shootReward = self.rewards[row, x_pos]  # the ship has hit an invader
                self.rewards[row, x_pos] = 0.0
                return self.shootReward
            else:
                self.shoot((row - 1))

    # check whether an invader is in front of the space ship
    def is_invader_ahead(self, row, state):
        x_pos = state[1]
        if row == -1:  # end of the grid
            return False
        else:
            if self.rewards[row, x_pos] == 2.0:
                return True
            else:
                return self.is_invader_ahead((row- 1),state)

    # get the correct reward for the step action
    def reward(self):
        r, c, l = self.current_state
        if self.shootReward != 0:  # shoot reward, is equal to zero if you take a move action
            return self.shootReward
        elif self.hit_board_reward != 0: #hit the board boundaries, get "reward penalization"
            return self.hit_board_reward
        else:  # move reward
            return self.rewards[r, c]

    def transition(self, state, a):
        action = self.actions[a]
        if a == 2:   # the shoot action has been chosen
            self.shootReward = - 1  # case where you have shot but not invaders have been hit
            self.shoot(action['row']) # call the shoot function
            return self.current_state
        else:   # move actions
            new_c = state[1] + action[1]
            # check if new position is out of bounds
            if (new_c >= 0) and (new_c <= self.cols - 1):
                return (state[0], new_c, state[2])
            else: # reward "penalization" if you hit the boundaries
                self.hit_board_reward = - 3
                return state

    def step(self, a_idx):
        self.shootReward = 0   # in every step the actual reward of the shoot action & hit board is resetted
        self.hit_board_reward = 0

        chosen_action = a_idx + np.random.choice([1, 0, -1], p=self.transition_probabilities)   # choose the action
        n_actions = 3
        # module the index of the action
        effective_action = chosen_action + n_actions if chosen_action < 0 else chosen_action % n_actions

        # do the transition and return all the info of the step

        prev_state = self.getCurrentState()
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



