from criticNet import CriticNetwork
from actorNet import ActorNetwork
from valueNet import ValueNetwork
from replayBuffer import ReplayBuffer
import os
import torch as T
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F
import itertools
from torch.optim import Adam

class SoftActorCriticAgent():
    def __init__(self, alpha = 0.2, input_dims = None, env = None, gamma= 0.99, n_actions = None,
                 max_size= 100000, tau = 0.005, layer_size = 256, layer2_size=256, batch_size= 32, reward_scale= 2, polyak=0.995, lr= 1e-3):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha  # Definition of the temperature parameter
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.polyak = polyak
        self.lossPi = []
        self.lossQ = []
        self.lr = lr

        """ 2 critics, minimum od the evalution of this state for these 2 networks in the calculation of the loss function
              by the critic part"""

        self.actor = ActorNetwork(input_dims, n_actions=n_actions,
                                  name='actor', max_action=env.action_space.high)
        self.critic_1 = CriticNetwork(input_dims, n_actions=n_actions,
                                      name='critic_1')
        self.critic_2 = CriticNetwork(input_dims, n_actions=n_actions,
                                      name='critic_2')
        self.value = ValueNetwork(input_dims, name='value_net')

        self.target_value = ValueNetwork(input_dims, name="target_net")

        # At this point i initialize the main V-network and the target V-network with the same parameters.
        for target_parameter, parameter in zip(self.target_value.parameters(), self.value.parameters()):
            target_parameter.data.copy_(parameter.data)

        self.value_network_optimizer = Adam(self.value.parameters(), lr=self.lr)
        self.q1_network_optimizer = Adam(self.critic_1.parameters(), lr=self.lr)
        self.q2_network_optimizer = Adam(self.critic_2.parameters(), lr=self.lr)
        self.policy_network_optimizer = Adam(self.actor.parameters(), lr=self.lr)



    def choose_action(self, state):
        # state = T.Tensor([state])
        #with T.no_grad():
        actions = self.actor.sample_normal(T.FloatTensor(state))#, dtype=T.float32))
    #    actions = actions.unsqueeze(0)
       # print(actions)
        action= actions.detach().cpu().numpy()
    #print(action)
       # action= action.numpy()[0]
       # print(action)
        return  action # transform from tensor to array and return               #actions.cpu().detach().numpy()[0]


    def save(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def sample_buffer_replay(self, batch_size):
        return self.memory.sample_buffer(batch_size)

    def learn(self, data):
        # take the data
        state = T.FloatTensor(data["states"])
        action = T.FloatTensor(data["actions"])
        reward = T.FloatTensor(data["rewards"]).unsqueeze(1)
        new_state = T.FloatTensor(data["states_"])
        done = T.FloatTensor(data["dones"]).unsqueeze(1)

        # I execute the forward for getting a prediction values.
        q1_prediction = self.critic_1.forward(state,action)
        q2_prediction = self.critic_2.forward(state, action)
        v_predicion = self.value.forward(state)


        # take the action from the policy
        actionPi, logP = self.actor.sample_normal_batch(state)

        ###########################################################################################    Training Q   ####

        target_value= self.target_value.forward(new_state)

        target_q_value =  reward  + self.gamma * (1 - done) * (target_value) #- self.alpha *logP)
      # print(target_q_value)
      # print(type(target_q_value))
      # print(target_q_value.detach())
      # print(type(target_q_value))
        target_q_value = target_q_value.detach()
       #print(q1_prediction.shape)
       #print(target_q_value.shape)
        q_loss1 = nn.MSELoss()(q1_prediction, target_q_value)
        q_loss2 = nn.MSELoss()(q2_prediction, target_q_value)



        self.q1_network_optimizer.zero_grad()
        q_loss1.backward()
        self.q1_network_optimizer.step()

        self.q2_network_optimizer.zero_grad()
        q_loss2.backward()
        self.q2_network_optimizer.step()

       # print("q_loss1" + str(q_loss1))
       # print("q_loss2" + str(q_loss2))

        ############################################################################################    Training V  ####

        new_q_value_prediction = T.min(self.critic_1.forward(state,actionPi), self.critic_2.forward(state,actionPi))
        new_q_value_prediction = new_q_value_prediction.squeeze()
        function_target_v = new_q_value_prediction - (logP * self.alpha)
        function_target_v = function_target_v.detach()
        v_predicion = v_predicion.squeeze()

        v_loss = nn.MSELoss()(v_predicion, function_target_v)



        self.value_network_optimizer.zero_grad()
        v_loss.backward()
        self.value_network_optimizer.step()

       # print("v_loss" + str(v_loss))


        ###########################################################################################    Training Pi  ####

        policy_loss = (logP * self.alpha)- new_q_value_prediction
        policy_loss = policy_loss.mean()


        self.policy_network_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_network_optimizer.step()

      #  print("policy_loss" + str(policy_loss))

        ##########################################################################################   Update target  ####

        for target_parameter, parameter in zip(self.target_value.parameters(), self.value.parameters()):
            target_parameter.data.copy_(  # the underscore replace the results in data
                target_parameter.data * (1.0 - self.tau) + parameter.data * self.tau
            )

    ######################################################################   save and load  ############################
        # Save model parameters
    def save_model(self, env_name):
        if not os.path.exists('models/'):
            os.makedirs('models/')
        actor_path = "models/actor__{}".format(env_name)
        critic1_path = "models/critic1__{}".format(env_name)
        critic2_path = "models/critic2__{}".format(env_name)
        value_path = "models/value__{}".format(env_name)
        target_value_path = "models/target_value__{}".format(env_name)
        print('Saving models to: \n - {}\n - {}\n - {}\n - {}\n - {}'.format(actor_path,
                                                                             critic1_path,
                                                                             critic2_path,
                                                                             value_path,
                                                                             target_value_path))
        T.save(self.actor.state_dict(), actor_path)
        T.save(self.critic_1.state_dict(), critic1_path)
        T.save(self.critic_2.state_dict(), critic2_path)
        T.save(self.value.state_dict(), value_path)
        T.save(self.target_value.state_dict(), target_value_path)

    # Load model parameters
    def load_model(self, actor_path, critic1_path, critic2_path, value_path,
                   target_value_path):
        print('Loading models from : \n - {}\n - {}\n - {}\n - {}\n - {}'.format(actor_path,
                                                                                 critic1_path,
                                                                                 critic2_path,
                                                                                 value_path,
                                                                                 target_value_path))
        if actor_path is not None:
            self.actor.load_state_dict(T.load(actor_path))
        if critic1_path is not None:
            self.critic_1.load_state_dict(T.load(critic1_path))
        if critic2_path is not None:
            self.critic_2.load_state_dict(T.load(critic2_path))
        if value_path is not None:
            self.value.load_state_dict(T.load(value_path))
        if target_value_path is not None:
            self.target_value.load_state_dict(T.load(target_value_path))
