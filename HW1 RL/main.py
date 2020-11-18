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
import time
from copy import deepcopy
import gym
import numpy as np
from gym.envs.registration import register

# random execution of the environment with 10 actions
def random_test(space_env):

    print("************Initial environment***************************\n")
    space_env.render()
    print(space_env.getCurrentState())
    print(space_env.reward())
    print(space_env.action_semantics)


    print("\n***********10 random actions********************************\n")


    for i in range(10):
        # print the transaction
        print("\n**********************"+ " iteration number: "+ str(i+1) + "*************************************\n" )
        observation, reward, done, info =space_env.step(space_env.action_space.sample())
        print("observation: " + str(observation))
        print("reward: " +str(reward))
        print("game complete: " + str(done))
        print("sampled action: " + space_env.action_semantics[info["sampled_action"]])
        print("effective action: " + space_env.action_semantics[info["effective_action"]])
        print("has invader/s ahead: " + str(space_env.is_invader_ahead(4, observation)))
        space_env.render()

        if done:
            break

    space_env.reset()


# function which computes the Value function
def policyEvaluation(pi, V, env, theta=0.1, discount = 0.8):
    state_space_size = env.cols # * 2
    while True:
        delta = 0 # variable used to track the updates
        for s in range(state_space_size):
            # initialization of v to zero then updated
            v = 0
            for act,action_prob in enumerate(pi[s]):
                # get the actual state of the environment (invaders disposition) and place the space ship in s
                env.setCurrentState((env.getCurrentState()[0], s, env.getCurrentState()[2]))

                # do the step and get results
                observation, reward, done, info = env.step(act)
                transition_prob = env.transition_probabilities[1]  # the chosen action index + 0

                # update v
                v += action_prob * transition_prob * (reward + discount* (V[observation[1]]))
            # take the max variation
            delta = max(delta, np.abs(v - V[s]))
            # update for the state
            V[s]= v
        # condition to leave the loop
        if delta < theta:
            break
    return V


def policyIteration(env, discount = 0.8):
    # retrieve state space and actions for the environment
    state_space_size = env.cols
    actions = env.action_semantics

    # vector of utilities per states in S initially zero
    V = np.zeros(state_space_size)

    # initialization of the policy, with an equal probability for all actions in states
    pi = np.ones([state_space_size, len(actions)]) / len(actions)
    # counting the number of step. useful for debugging
    step = 0
    # main loop
    while True:
        # update the step
        step += 1
        # compute the Value function
        V = policyEvaluation(pi, V, env)
        # get the transition probability
        transition_prob = env.transition_probabilities[1]
        # Variable to check if there's an update of the choosen action for the state,setted to false if not
        policy_stable = True
        for s in range(state_space_size):

            # choose the best action fot the current policy
            old_action = np.argmax(pi[s])
            # definition of an array for the policy values
            acts_values = np.zeros(len(actions))
            for a in range (len(actions)):
                # set up of the environment
                env.reset()
                env.setCurrentState((env.getCurrentState()[0], s, env.getCurrentState()[2]))

                # do the step and get results
                observation, reward, done, info = env.step(a)
                # compute the policy values
                acts_values[a] += transition_prob * (reward + discount * V[observation[1]])

            # choose the new action as the best among them
            new_action = np.argmax(acts_values)

            # check for updates in the actions
            if old_action != new_action:
                policy_stable = False
            # update the policy for this state
            pi[s]=  np.eye(len(actions))[new_action] # acts_values

            # if you complete the game go to the next state evaluation
            if done:
                break
        # if no update, return the policy
        if policy_stable:
            env.reset()
            return pi



# ************************** Booting *******************************

# function to give a look for the results of the policy evaluation
def testPolicyEvaluation(env):
    state_space_size = env.cols
    actions = env.action_semantics
    V = np.zeros(state_space_size)
    pi = np.ones([state_space_size, len(actions)]) / len(actions)
    V = policyEvaluation(pi, V, env)
    print(V)
# function to give a look for the results of the policy iteration
def testPolicyIteration(env):
    pi, V = policyIteration(env)
    print("Policy: " + str(pi))

# function which test the policy
def policyTest(env, testType = "on_environment"):
    # reset the environment before test
    env.reset()
    if testType == "policy_eval":
        testPolicyEvaluation(env)
    elif testType == "policy_iter":
        testPolicyIteration(env)
    # test on_your environment
    else:
        # to compute the policy has been done this reasoning:
        # take a reduce space state (only thinking about the space ship possible positions, hence 7 states,
        # instead of 2^35 * 7 possible states considering all the possible invaders permutations. A new policy will be
        # compute for each loop passing the actual state with the configuration of invaders.
        # brief consideration: the procedure take enough time for each step, the results are not allways good, the
        # choosen action by the policy sometimes is improper, i wrote the algorithm following the instruction on the
        # book "RLbook2018".

        for i in range(5):
            # make an image of the environment before the policy evaluation
            state = deepcopy(env.current_state)
            reward = deepcopy(env.rewards)

            # policy calculus
            policy = policyIteration(env)
            time.sleep(1)
            print("****Policy****:\n")
            print(policy)

            # restore the image of the environment
            env.setCurrentState(state)
            env.setRewards(reward)

            # act following the policy, print section
            print("\n**********************" + " iteration number: " + str(i + 1) + "*************************************\n")
            x_pos = env.getCurrentState()[1]
            action = np.argmax(policy[x_pos])
            observation, reward, done, info = env.step(action)
            print("observation: " + str(observation))
            print("reward: " + str(reward))
            print("game complete: " + str(done))
            print("choosen action: " + env.action_semantics[info["sampled_action"]])
            print("effective action: " + env.action_semantics[info["effective_action"]])
            print("has invader/s ahead: " + str(env.is_invader_ahead(4, observation)))
            env.render()

            # if complete exit from the loop
            if done:
                break

if __name__ == "__main__":
    register(
        id='spaceInvaders-v0',
        entry_point='gym_game.env:SpaceInvadersEnv',
        max_episode_steps=10000,
    )
    # creation of the gum environment
    env = gym.make('spaceInvaders-v0')
    env.reset()

    print("*************************************************************************************")
    print("********************************* Random test ***************************************")
    print("*************************************************************************************")
    # first task of the homework
    random_test(env)


    print("*************************************************************************************")
    print("********************************* Policy test ***************************************")
    print("*************************************************************************************")
    # second task of the homework
    policyTest(env,testType= "on_environment")



