"""
Reinforcement learning Homework n°1
Problem: "SpaceInvaders-v0"
features: Game_grid -> 7 * 7, static invaders
Representation of the problem
x-> space_ship
i->invaders

    0   1   2   3   4   5   6
0   i   i   i   i   i   i   i
1   i   i   i   i   i   i   i
2   i   i   i   i   i   i   i
3   i   i   i   i   i   i   i
4   i   i   i   i   i   i   i
5   0   0   0   0   0   0   0
6   0   0   0   x   0   0   0

"""
from typing import Dict, List, Tuple

import dataclasses
import numpy as np
import gym
from gym import spaces


# ********************************************** MDP *******************************************************************

@dataclasses.dataclass
class Transition:
    state: Tuple[int, int, np.array]
    action: str
    next_state: Tuple[int, int, np.array]
    reward: float
    termination: bool


class Space_env(gym.Env):

    """Custom Environment that follows gym interface"""

    _game = "SpaceInvaders-v0"
    _states: np.array
    _rewards: np.array
    _action_semantics: List[str]
    _actions: Dict[str, np.array]
    _init_state: Tuple[int, int, np.array]   # x, y, invaders
    _current_state: Tuple[int, int, np.array]
    _goal: int  # numbers of alive invaders
    _transition_probabilities: np.array

    def __init__(self,
                 rows: int,
                 cols: int,
                 step_cost: float,
                 invaders: np.array = None,
                 goal = int) -> None:

        super(Space_env, self).__init__()
        self.cols = cols
        self.rows = rows
        self._goal = goal


        # create rewards table
        self._rewards = np.zeros((rows,cols))
        self._rewards[rows-1] = -step_cost    # step cost of moves
        self._shootReward = 0
        for r, c in invaders:
            self._rewards[r,c] = + 0.5  # reward gained for the elimination of invaders

        # initialization of the matrix of states and definition of the current state
        self._states = np.zeros((rows, cols))
        self._init_state = (rows - 1, 3, invaders)
        self._current_state = self._init_state


        # handle of the agent's actions
        self._action_semantics = ['left', 'right', 'shoot']
        self.shoot_property={'row': 4}
        self._actions = np.array([[0, -1], [0, 1], self.shoot_property])  # shoot dictionary for the activation of the shoot func.

        # gym attribute
        self.action_space = spaces.Discrete(len(self._actions))  # allowed action non-negative numbers (0,1,2)
        high = 1; low = 0
        self.observation_space = spaces.Box(low, high, shape=(7, 7))

        #define the transition model P(s'|s,a)
        self._transition_probabilities = np.array([0.1,0.8,0.1])


    def shoot(self, row):
        x_pos= self._current_state[1]
        if row == -1 :  # end of the grid
            return self._shootReward
        else:
            if self._rewards[row,x_pos] == 0.5:
                r, c, invaders = self._current_state
                invaders.remove((row, x_pos))
                self._shootReward= self._rewards[row, x_pos] # the ship has hit an invader
                self._rewards[row, x_pos] = 0.0
                return  self._shootReward
            else:
                self.shoot((row - 1))

    @property
    def actions(self) -> List[str]:
        return self._action_semantics

    @property
    def current_state(self) -> Tuple[int, int, np.array]:
        return self._current_state

    @property
    def reward(self) -> float:
        r, c, l = self._current_state
        if self._shootReward != 0:     # shoot reward
            return self._shootReward
        else:                          # move reward
            return self._rewards[r, c]

    @property
    def termination(self) -> bool:
        r, l, c = self._current_state
        return len(c) == self._goal   # killed all the invaders

    def render(self) -> None:
        grid = np.array(self._states, dtype=str)
        r, c, invaders = self._current_state
        grid[r, c] = ' X '

        for x, y in invaders:
            grid[x, y] = ' I '

        print(grid)

    def _transition(self, state:  Tuple[int, int, np.array], a: int) -> Tuple[int, int, np.array]:
        n_actions = len(self._actions)
        a_module = a + n_actions if a < 0 else a % n_actions
        action = self._actions[a_module]  # module the index of the action
        if a_module == 2:   # the shoot action has been chosen
            self.shoot(action['row'])
            return self._current_state
        else:
            new_c = state[1] + action[1]
            # check if new position is out of bounds
            return (state[0], new_c, state[2]) if (new_c >= 0) and (new_c <= self.cols-1) else state


    def step(self, action: str) -> Transition:
        self._shootReward = 0   # in every step the actual reward of the shoot action is resetted

        a_idx = self._action_semantics.index(action)
        chosen_action = a_idx + np.random.choice([1, 0, -1], p=self._transition_probabilities)   #probability of doing that action
        prev_state = self._current_state
        self._current_state = self._transition(self._current_state, chosen_action)
        return Transition(state=prev_state,
                          action=action,
                          next_state=self._current_state,
                          reward=self.reward,
                          termination=self.termination)

    def reset(self) -> None:
        self._current_state = self._init_state

    def state_space_size(self) -> Tuple[int, int]:
        return self._states.shape

    def action_space_size(self) -> int:
        return len(self._actions)

# ***************************************** Policy iteration algorithm *************************************************


def policyEvaluation(pi, U, env, theta=0.00001):
    state_space_size = env.state_space_size()[0]
    actions = env.actions

    while True:
        delta = 0 # to track the updates
        for s in range(state_space_size):
            v = 0
            env._current_state = (env._current_state[0], s, env._current_state[2])
            num_action = 0

            for adict in (pi[s]):
                print(a)
                transition = env.step(a)
                action_prob = env._transition_probabilities[num_action]
                num_action += 1
                v += policy_prob * action_prob * (transition.reward + U[transition.next_state[1]])
            delta = max(delta, np.abs(v - U[s][a]))
            U[s][a] = v
            if delta < theta:
                break
            return U



def policyIteration(env):
    state_space_size = env.state_space_size()[0]  #our agent can move itself only in the last raw

    actions = env.actions
    print(state_space_size)
    print(actions)
    gamma = 0.99
    epsilon = 0.1

    # policy function, compute the policy from a q function
    def policy(eps: float, actions: List[str]) -> Dict[int, str]:
        # initialize the vector of utility with no values
        U: Dict[int, Dict[str, List[float]]] = {i: {a: 0 for a in actions}
                                                for i in range(state_space_size)}

        # epsilon greedy policy
        pi = {i: actions[np.argmax(U[i])] if np.random.rand() >= eps else actions[np.random.randint(len(actions))]
              for i in U.keys()} # i-> state
        return pi , U

    pi, U = policy(epsilon,actions)
    print(pi)
    print(U)
    policyEvaluation(pi, U, env)


# ***************************************** Booting  *******************************************************************
def test():
    invs= [] # creation of invaders
    for a in range(5):
        for b in range(7):
            invs.append((a,b))

    space_env = Space_env (rows=7, cols=7, step_cost=1, goal= 0, invaders=invs)

    print("************Initial environment***************************\n")
    space_env.render()
    print(space_env.current_state)
    print(space_env.reward)
    print(space_env.termination)
    print(space_env.actions)
    print("\n***********5 random actions********************************\n")

    for _ in range(0):
        choice = np.random.choice(space_env.actions)
        print(choice)
        print(space_env.step(choice))  # print the transaction
        space_env.render()

    space_env.reset()

    policyIteration(space_env)
test()


