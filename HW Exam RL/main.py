"""
******* Project specifications
env: FetchReach-v1
aim: A goal position is randomly chosen in 3D space. Control Fetch's end effector to reach that goal as quickly as possible.
framework: pytorch
reference: paper "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
          by Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine

steps for the algorithm:
1 -> definition of a class for the experience replay, a buffer to store data, and sampling batches
2 -> implementation of sac:
    2.1 Q function loss -> computed by taking the data from exp replay,
    compute the q value of the observation and the action according to 2 q functions (2 neural networks for it)
    and then we use the minimum, then we have to move the gradient computation cause when compute the target we don't want
    gradient flow through the target but only the other part of the equation where we're computing the difference, the mean square error
    from the target, we're using target q value q1 and target q value q2, tha action (a2) is sampled from the policy, and then we take the minimum
    of q1 and q2 target. and finally the target (backup). then the loss for the q functions is q - the target, all this under the torch.no_grad(),
    otherwise the things will be broken, we want the gradient action only for q1 and q2, so the prediction in the current state and current action


{'observation': array([ 1.34183265e+00,  7.49100387e-01,  5.34722720e-01,  1.97805133e-04,
        7.15193042e-05,  7.73933014e-06,  5.51992816e-08, -2.42927453e-06,
        4.73325650e-06, -2.28455228e-06]), 'achieved_goal': array([1.34183265, 0.74910039, 0.53472272]), 'desired_goal': array([1.29535769, 0.79181199, 0.49400808])}
"""

# imports
import gym
import sys
import numpy as np
from sacAgent import SoftActorCriticAgent
import matplotlib.pyplot as plt




def main_loop():
  # dictionary with execution information
  exe_info = {}

  # init configuation variables
  n_episode = 50#300
  max_steps = 50#200
  max_start_step= 1000   # after this, start using the agent's policy
  learnStartStep = 500 # after this, start the learning update process
  stepInBatch = 32
  batch_size = 32
  HER_StartEpisode = 1 # after this, start using HER whether enabled
  HER_GoalsStep_n = 10 # after this number of steps, sample a goal
  HER_enabled = False
  render_on = True
  random_action = False
  # end configuation variables

  def reShape(s):
    obs = np.reshape(s['observation'], (1, 10))
    ach = np.reshape(s['achieved_goal'], (1, 3))
    des = np.reshape(s['desired_goal'], (1, 3))
    state = np.concatenate([obs, des], axis=1)
    return list(state.squeeze())

  def reShapeHer(s, goal=[]):
    try:
      goal = np.reshape(goal, (1, 3))
    except:
      goal  = np.reshape(s['desired_goal'], (1, 3))

    obs = np.reshape(s['observation'], (1, 10))
    ach = np.reshape(s['achieved_goal'], (1, 3))
    state = np.concatenate([obs, goal], axis=1)

    return list(state.squeeze())

  # ******************************************* setting the environment ***********************************************

  env = gym.make('FetchReach-v1')
  print("action space: " + str(env.action_space))
  print("action space low : " + str(env.action_space.low))
  print("action space high : " + str(env.action_space.high))
  print("observation space :" + str(env.observation_space))
  env._max_episode_steps = max_steps

  num_inputs = env.observation_space['observation'].shape[0] + env.observation_space['desired_goal'].shape[0]  # interesting observations are the Cartesian coordinates of the gripper
  num_action = env.action_space.shape[0]  # env.action_space.shape[0] -> : 3 dimensions specify the desired gripper movement in Cartesian coordinates and
  # the last dimension controls opening and closing of the gripper (1 action can be avoided for this task (the gripper one)
  # sets an initial state
  env.reset()
  # setting the agent
  agent = SoftActorCriticAgent(env = env, input_dims = (13,), n_actions= num_action)

  # ***************************************************** evaluations initiliazation **********************************
  # score evaluation (can do the same for the number of steps: https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce-f887949bd63
  best_score = env.reward_range[0]
  score_pool= []
  avg_score5ep = []
  n_steps = []  # step value when the agent reach the goal, hence, the actuator is in the goal position
  avg_n_steps = []
  # **************************************************** end evaluations initiliazation ********************************
  start_step = 0
  # start episodes


  for ep in range(n_episode):
    state = env.reset()    #state==observation
    done = False
    # store the rewards for each episode
    scores = []
    # boolean variable which stores if the goal has been reached in the episode
    goal_reached = False
    #store transtions and goals for HER
    stepTransitions = []
    goals = []

    # second loop for steps in episode
    for steps in range(max_steps):
      start_step += 1
      if render_on:       # renders the environment
        env.render()
      if random_action:
        action = env.action_space.sample()     # Takes a random action from its action space
      else: # use the agent
        if start_step >= max_start_step:
          action = agent.choose_action(reShape(state))
        else:
          action = env.action_space.sample()
      new_state, reward, done, info = env.step(action)
      if False:
        print("*************")
        print(state)
        print(new_state)
        print("*************")

      # save in buffer
      if not random_action:
        agent.save(reShape(state), action, reward, reShape(new_state), done)

        if not((steps+1) + HER_GoalsStep_n > max_steps):
          data = (state,action,reward,new_state)
          stepTransitions.append(data)
        if (steps+1)%HER_GoalsStep_n == 0 and steps !=0:
          goals.append(new_state['achieved_goal'])
       # print(steps)
       # print(new_state['achieved_goal'])

      # save data from the execution
      scores.append(reward)
      state = new_state
      if reward == 0 and not(goal_reached): #in the goal position
        goal_reached = True # i store only the first i reach the goal position
        n_steps.append(steps)
        avg_n_steps.append(np.mean(n_steps[-10:]))  # mean of the last 10 values

      # updates, learn the agent
      if start_step >= learnStartStep and start_step % stepInBatch == 0:
        for _ in range(stepInBatch):
        # states, actions, rewards, nextStates, dones = agent.sample_buffer_replay(batch_size)
          data = agent.sample_buffer_replay(batch_size)
          #print("Updating the model...")
          agent.learn(data)  #states,actions,rewards,nextStates,dones)

      if done:
        # save data from the execution
        if not(goal_reached):
          n_steps.append(steps)
          avg_n_steps.append(np.mean(n_steps[-10:]))  # mean of the last 10 values
        score_pool.append(np.sum(scores))
        if score_pool:
          avg_score5ep.append(np.mean(score_pool[-(max_steps * 5):]))  # mean of the last

        # print data from the episode
        print("Episode: " +str(ep) + " | Total reward "+ str(np.round(np.sum(scores))) + " | average_reward: " +
              str(np.round(np.mean(score_pool[-10:]))) + " | Step number: " + str(steps+1))
        """
        if avg_score > best_score:
          best_score= avg_score
          print("best score: " + str(best_score)) """
        if np.sum(scores) > best_score:
          best_score= np.sum(scores)
          print("best score: " + str(best_score))
        # conclude the episode (useless in this case (?))
        break
   # print(goals)
   # print(state['achieved_goal'])
   # print(len(goals))

    # ********** start HER ****************************
    if HER_enabled and ep>= HER_StartEpisode:
      print("Using Hindsight experience replay...")
      # ********HER*************************
      # If we want, we can substitute a goal here and re-compute
      # the reward. For instance, we can just pretend that the desired
      # goal was what we achieved all along.

      # the goal is the state that we have in the last step of the episode
      substitute_goal = state['achieved_goal'].copy()

      i = 0
      for iter in range(len(stepTransitions)):
        transition = stepTransitions[iter]

        # get the max reward for staying in the desired position
        substitute_reward = env.compute_reward( state['achieved_goal'], substitute_goal, info)
        if (iter)%HER_GoalsStep_n==0: #and iter != 0:
          goal = goals[i]
          i+=1
        if i==0:
          goal = []
          reward = -1
        substitute_reward = env.compute_reward(transition[3]['achieved_goal'], goal, info)
        Done = (substitute_reward == 0.0)

        if False:
          print("dfdsf")
          print(transition[0])
          print(goal)
          print("dfdsf")
          print(Done)
          print((reShapeHer(transition[0],goal), transition[1], substitute_reward, reShapeHer(transition[3],goal),Done))
          print(substitute_reward)

        agent.save(reShapeHer(transition[0],goal), transition[1], substitute_reward, reShapeHer(transition[3],goal), True)
        # ****************** end HER *************************

  # store information in a dictionary
  exe_info['best_score'] = best_score + max_steps

  exe_info['scores_history'] = list(np.asarray(score_pool)+  max_steps)
  exe_info["avgScore5ep"] =list( np.asarray(avg_score5ep) + max_steps)

  exe_info['nstep_history'] = n_steps
  exe_info['avg10step'] = avg_n_steps

  # ending the episodes i close the environment and return execution data
  env.close()
  return exe_info

def plot_results(info):
  # plot statements

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
  print(info)

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

  #idea for plots: 2 * n, with number of schemas, 1 with her and the other without


if __name__ == "__main__":
  info = main_loop()
 # plot_results(info)
