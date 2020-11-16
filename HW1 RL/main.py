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

import gym
from gym.envs.registration import register

def test(space_env):

    """
    print("************Initial environment***************************\n")
    space_env.render()
    print(space_env.current_state)
    print(space_env.reward())
    print(space_env.termination())
    print(space_env.action_semantics)
    """

    print("\n***********5 random actions********************************\n")


    for i in range(1000):
        # print the transaction
        print("\n**********************"+ " iteration number: "+ str(i+1) + "*************************************\n" )
        observation, reward, done, info =space_env.step(space_env.action_space.sample())
        print("observation: " + str(observation))
        print("reward: " +str(reward))
        print("game complete: " + str(done))
        print("sampled action: " + env.action_semantics[info["sampled_action"]])
        print("effective action: " + env.action_semantics[info["effective_action"]])
        print("prev state: " + str(info["prev_state"]))
        space_env.render()

        if done:
            break

    space_env.reset()



if __name__ == "__main__":
    register(
        id='spaceInvaders-v0',
        entry_point='gym_game.env:SpaceInvadersEnv',
        max_episode_steps=2000,
    )
    env = gym.make('spaceInvaders-v0')
    env.reset()

    MAX_EPISODES = 9999
    MAX_TRY = 1000
    epsilon = 1
    epsilon_decay = 0.999
    learning_rate = 0.1
    gamma = 0.6

    # first task of the homework
    test(env)

    # second task of the homework


