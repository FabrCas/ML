"""
******* Project specifications ******
env: FetchReach-v1
aim: A goal position is randomly chosen in 3D space. Control Fetch's end effector to reach that goal as quickly as possible.
framework: pytorch
reference: paper "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
          by Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine

observation form example: {'observation': array([ 1.34183265e+00,  7.49100387e-01,  5.34722720e-01,  1.97805133e-04,
                                          7.15193042e-05,  7.73933014e-06,  5.51992816e-08, -2.42927453e-06,
                                          4.73325650e-06, -2.28455228e-06]),
                  'achieved_goal': array([1.34183265, 0.74910039, 0.53472272]),
                  'desired_goal': array([1.29535769, 0.79181199, 0.49400808])}
"""

# *********************************************** start import modules  ************************************************
import gym
import numpy as np
from sacAgent import  SoftActorCriticAgent
import matplotlib.pyplot as plt
# *********************************************** end import modules  **************************************************


# *********************************************** init configuration variables *****************************************
n_episode = 20
max_steps = 400
max_start_step= 1000 # after this, start using the agent's policy
learnStartStep = 400  #1500 #500 # after this, start the learning update process
stepInBatch = 32  # every how much steps doing the update of params
batch_size = 32
HER_StartEpisode = 1 # after this, start using HER whether enabled
HER_GoalsStep_n = 10 # after this number of steps, sample a goal
HER_enabled = True
render_on = False
random_action = False
# ********************************************** end configuration variables *******************************************

# *********************************************** start aux functions **************************************************
# the plotting function
def plot_results(info):

  # plot for steps number
  """
  plt.plot(info['nstep_history'])
  plt.plot(info['avg10step'])
  plt.xlabel('Episodes')
  plt.ylabel("Steps")
  plt.show()"""

  "plot for rewards"
  plt.plot(info['scores_history'])
  plt.xlabel('Episodes')
  plt.ylabel("Rewards")
  plt.show()

  "plot step for average return"
  plt.plot(info["nstep_history"])
  plt.plot(info["avg10step"])
  plt.xlabel('Episodes')
  plt.ylabel("Steps to the reach goal position")
  plt.show()

  "plot step for average return"
  plt.plot(info["avgScore5ep"])
  plt.xlabel('Episodes')
  plt.ylabel("Average Rewards")
  plt.show()

  # plot loss values networks

  "plot the loss Q"
  plt.plot(info["lossPi"])
  plt.xlabel('Steps')
  plt.ylabel("loss Policy")
  plt.show()

  "plot the loss V"
  plt.plot(info["lossV"])
  plt.xlabel('Steps')
  plt.ylabel("loss V")
  plt.show()

  "plot the loss Pi"
  plt.plot(info["lossQ"])
  plt.xlabel('Steps')
  plt.ylabel("loss Q")
  plt.show()

# reshape functions

# reshape for step observations
def reShape(s):
  obs = np.reshape(s['observation'], (1, 10))
  ach = np.reshape(s['achieved_goal'], (1, 3))
  des = np.reshape(s['desired_goal'], (1, 3))
  state = np.concatenate([ach, des], axis=1)
  return list(state.squeeze())

# reshape for HER observations
def reShapeHer(s, goal=[]):
  try:
    goal = np.reshape(goal, (1, 3))
  except:
    goal  = np.reshape(s['desired_goal'], (1, 3))
  obs = np.reshape(s['observation'], (1, 10))
  ach = np.reshape(s['achieved_goal'], (1, 3))
  state = np.concatenate([ach, goal], axis=1)
  return list(state.squeeze())

# *********************************************** end aux functions ****************************************************


# ******************************************* start  main loop function ************************************************

def main_loop():
  # dictionary with execution information
  exe_info = {}

  # ******************************************* setting the environment ************************************************

  # definition of the environment and its maximum number of step per epoch
  name_env = 'FetchReach-v1'
  env = gym.make(name_env)
  env._max_episode_steps = max_steps

  # print statements about action and observation space
  print("action space: " + str(env.action_space))
  print("action space low : " + str(env.action_space.low))
  print("action space high : " + str(env.action_space.high))
  print("observation space :" + str(env.observation_space))

  # evaluating the number of inputs, after reshape, and the addition of the goal
  num_inputs = env.observation_space['achieved_goal'].shape[0] + env.observation_space['desired_goal'].shape[0]
  # evaluating the dimension of the continuous actions
  num_action = env.action_space.shape[0]
  # env.action_space.shape[0] -> : 3 dimensions specify the desired gripper movement in Cartesian coordinates and
  # the last dimension controls opening and closing of the gripper

  # sets an initial state
  env.reset()

  # print input after reshape
  print("shape of the input {}".format(num_inputs))

  # definition of the agent through the SAC module
  agent = SoftActorCriticAgent(env = env, input_dims = (6,), n_actions= num_action)
  # input_dims: NO RESHAPE -> (13,0) or RESHAPE -> (6,0)

  # ***************************************************** evaluations initiliazation ***********************************

  best_score = env.reward_range[0]
  score_pool= []
  avg_score5ep = []
  n_steps = []  # step value when the agent reach the goal, hence, the actuator is in the goal position
  avg_n_steps = []

  # **************************************************** end evaluations initiliazation ********************************
  start_step = 0
  agent_started = False

  # start episodes


  for ep in range(n_episode):

    # take the first observation from the reset
    state = env.reset()

    # array to store the rewards for each episode
    scores = []

    # boolean variable which stores if the goal has been reached in the episode, used to compute plots about the "speed"
    goal_reached = False

    # arrays used to store transitions and goals for HER
    stepTransitions = []
    goals = []

    # second loop for steps in episode
    for steps in range(max_steps):
      start_step += 1

      # renders the environment through the graphic engine
      if render_on:
        env.render()

      # sample action, randomly or using the agent
      # Takes a random action from its action space
      if random_action:
        action = env.action_space.sample()

      # use the agent
      else:
        # after max_start_step, use the agent policy
        if start_step >= max_start_step:
          # print the change from random to policy
          if not(agent_started):
            print("*******************+ switched from stochastic actions to agent's choice ***************************")
            agent_started = True
          action = agent.choose_action(reShape(state))
        else: # still random action
          action = env.action_space.sample()

      #execute the action and take data
      new_state, reward, done, info = env.step(action)

      # save in buffer the transition
      agent.save(reShape(state), action, reward, reShape(new_state), done)

      # save the current transition used by HER
      if not((steps+1) + HER_GoalsStep_n > max_steps):
        data = (state,action,reward,new_state)
        stepTransitions.append(data)

      # sample a goal for HER
      if (steps+1)%HER_GoalsStep_n == 0 and steps !=0:
        goals.append(new_state['achieved_goal'])

      # save information about rewards
      scores.append(reward)

      # update the state
      state = new_state

      # save the step number that reaches the goal first
      if reward == 0 and not(goal_reached):
        goal_reached = True # reached, now i'm not interested in the other step of the episode

        # save it for the evaluation, with a variant that considers the average
        n_steps.append(steps)
        avg_n_steps.append(np.mean(n_steps[-10:]))  # mean of the last 10 values

      # if we have done enough steps, start learning every stepInBatch step
      if start_step >= learnStartStep and start_step % stepInBatch == 0:
        # iteration for the stepInBatch value
        for _ in range(stepInBatch):

          # take data from buffer
          data = agent.sample_buffer_replay(batch_size)

          # update the parameters of the networks
          agent.learn(data)

      # episode ended, store other data, and print the overall reward
      if done:
        # save data from the execution

        # goal not reached, hence, we have to store the worst result
        if not(goal_reached):
          n_steps.append(steps)
          avg_n_steps.append(np.mean(n_steps[-10:]))  # mean of the last 10 values

        # save the global score of the episode
        score_pool.append(np.sum(scores))

        # save a score that takes the average of the last 5 step
        if score_pool:
          avg_score5ep.append(np.mean(score_pool[-(max_steps * 5):]))  # mean of the last

        # print data from the episode
        print("Episode: " +str(ep) + " | Total reward "+ str(np.round(np.sum(scores))) + " | average_reward: " +
              str(np.round(np.mean(score_pool[-10:]))) + " | Step number: " + str(steps+1))

        # highlight with a message if a best score has been achieved
        if np.sum(scores) > best_score:
          best_score= np.sum(scores)
          print("best score: " + str(best_score))

        # break in case of done used not only for the ending of the episode
        break


    # ********************************************** start HER *********************************************************
    if HER_enabled and ep>= HER_StartEpisode:
      print("Using Hindsight experience replay...")

      # If we want, we can substitute a goal here and re-compute
      # the reward. For instance, we can just pretend that the desired
      # goal was what we achieved all along.

      # the goal is the state that we have in the last step of the episode
      substitute_goal = state['achieved_goal'].copy()

      i = 0
      for iter in range(len(stepTransitions)):
        transition = stepTransitions[iter]


        substitute_reward = env.compute_reward( state['achieved_goal'], substitute_goal, info)
        # take goals for each transition in the last episode
        if (iter)%HER_GoalsStep_n==0:
          goal = goals[i]
          i+=1
        if i==0:
          goal = []
          reward = -1

        # get the max reward for staying in the desired position
        substitute_reward = env.compute_reward(transition[3]['achieved_goal'], goal, info)

        # reshape e save in the buffer the HER goal virtualized experience
        agent.save(reShapeHer(transition[0],goal), transition[1], substitute_reward,
                   reShapeHer(transition[3],goal), True)
    # ************************************************ end HER *********************************************************

  # store execution information in a dictionary

  exe_info['best_score'] = best_score + max_steps
  exe_info['scores_history'] = list(np.asarray(score_pool)+  max_steps)
  exe_info["avgScore5ep"] =list( np.asarray(avg_score5ep) + max_steps)
  exe_info['nstep_history'] = n_steps
  exe_info['avg10step'] = avg_n_steps
  exe_info['lossPi'] = agent.lossPi
  exe_info['lossQ'] = agent.lossQ
  exe_info['lossV'] = agent.lossV

  # save the model
  agent.saveModel(name_env)

  # ending the episodes i close the environment and return execution data
  env.close()
  return exe_info

# ******************************************** end  main loop function *************************************************


# **********************************************************************************************************************
# ************************************************* Main call **********************************************************
# **********************************************************************************************************************
if __name__ == "__main__":
  info = main_loop()
  plot_results(info)

