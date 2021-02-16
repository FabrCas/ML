from criticNet import CriticNetwork
from actorNet import ActorNetwork
from valueNet import ValueNetwork
from replayBuffer import ReplayBuffer
import saver_loader as os_mem
import torch as T
import torch.nn as nn
import  itertools
from torch import optim
""" Soft actor critic agent """
class SoftActorCriticAgent():
    def __init__(self, alpha = 0.2, input_dims = None, env = None, gamma= 0.99, n_actions = None,
                 max_size= 10000000, batch_size= 32, polyak=0.995, lr= 1e-3):
        self.gamma = gamma
        self.alpha = alpha  # Definition of the temperature parameter
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.polyak = polyak
        self.lossPi = []
        self.lossQ = []
        self.lossV = []
        self.lr = lr

        """ Definition of the neural networks: 1 actor, 2 critics, 1 value and 1 target value"""
        # 2 ways of acting for estimating the value function:
        # 1) define and learn a specific neural network (the used one)
        # 2) evaluate the value of a certain state V(st) from the expected values of the difference between
        # the q function Q(st, at) minus the entropy log(at|st), with at in reference to a certain policy pi

        self.actor = ActorNetwork(input_dims, n_actions=n_actions,
                                  name='actor', max_action=env.action_space.high)
        self.critic_1 = CriticNetwork(input_dims, n_actions=n_actions,
                                      name='critic_1')
        self.critic_2 = CriticNetwork(input_dims, n_actions=n_actions,
                                      name='critic_2')
        self.value = ValueNetwork(input_dims, name='value_net')

        self.target_value = ValueNetwork(input_dims, name="target_net")

        # Initialize the main V-network and the target V-network with the same parameters.
        for target_parameter, parameter in zip(self.target_value.parameters(), self.value.parameters()):
            target_parameter.data.copy_(parameter.data)

        # for simplicity i take the parameters of the Q networks and store in a unique variable
        CriticParameters = itertools.chain(self.critic_1.parameters(), self.critic_2.parameters())
        # define the Adam optimizer for those parameters
        self.optimizerCritic = optim.Adam(CriticParameters, lr=self.critic_1.lr)

    # function to sample the action
    def choose_action(self, state):
        # sample action from the actor normal
        actions = self.actor.sample_normal(T.FloatTensor(state))
        # transform from tensor to array and return, detach to carry out of the gradient
        action = actions.detach().cpu().numpy()
        return action

    # save in the experience Replay
    def save(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    # sample data from the experience replay
    def sample_buffer_replay(self, batch_size):
        return self.memory.sample_buffer(batch_size)

    # ***************************************** start loss functions ***************************************************
    def evalute_lossQ(self, reward, new_state, done, q1_prediction, q2_prediction,):

        target_value = self.target_value.forward(new_state)
        # in the paper the target value is estimated without the network using this formula:
        # (q_target - self.alpha * log_prob) = V(st), with q_target the min of 2 q target values
        target_q_value = reward + self.gamma * (1 - done) * (target_value)

        # we want gradient out of the target, which only effects the other part of the difference that we're going
        # to compute, hence the gradient flows only in q1 and q2
        target_q_value = target_q_value.detach()

        # loss through the mean square error
        q_loss1 = nn.MSELoss()(q1_prediction, target_q_value)
        q_loss2 = nn.MSELoss()(q2_prediction, target_q_value)

        # final value of the loss as the sum of the two q_loss
        q_loss = q_loss1 + q_loss2

        # store the value
        self.lossQ.append(q_loss)

        return  q_loss


    def evaluate_lossV(self, state, actionPi, entropy, v_prediction):

        # evaluate the Q values from the action given by Pi, and take the minimum
        new_q_value_prediction = T.min(self.critic_1.forward(state, actionPi), self.critic_2.forward(state, actionPi))
        new_q_value_prediction = new_q_value_prediction.squeeze()

        # the target from the q prediction and the entropy by Pi
        function_target_v = new_q_value_prediction - (entropy * self.alpha)
        function_target_v = function_target_v.detach()

        v_prediction = v_prediction.squeeze()

        # Mean square error to estimate the loss
        v_loss = nn.MSELoss()(v_prediction, function_target_v)

        # store the value
        self.lossV.append(v_loss)

        # return also the q_prediction using Action Pi, used to evaluate the Pi loss
        return v_loss, new_q_value_prediction

    def evaluate_lossPi(self, entropy, new_q_value_prediction):

        # Policy loss as the minimum q value by critics and the entropy (adjusted by the temperature parameter)
        policy_loss = (entropy * self.alpha) - new_q_value_prediction
        policy_loss = policy_loss.mean()

        # store the value
        self.lossPi.append(policy_loss)

        return  policy_loss

    # ***************************************** end loss functions ***************************************************

    # function to learn and update the parameters of the networks
    def learn(self, data):
        # extract information from data
        state = T.FloatTensor(data["states"])
        action = T.FloatTensor(data["actions"])
        reward = T.FloatTensor(data["rewards"]).unsqueeze(1)
        new_state = T.FloatTensor(data["states_"])
        done = T.FloatTensor(data["dones"]).unsqueeze(1)

        # I execute the forward for getting a prediction values of Q1,Q2, and V
        q1_prediction = self.critic_1.forward(state,action)
        q2_prediction = self.critic_2.forward(state, action)
        v_predicion = self.value.forward(state)


        # take the action from the policy
        actionPi, entropy = self.actor.sample_normal_batch(state)

        # compute the loss for each network and apply the stochastic gradient descend

        #******************************************************************************************    Training Q   ****

        q_loss = self.evalute_lossQ(reward, new_state, done, q1_prediction, q2_prediction)

        self.optimizerCritic.zero_grad()
        q_loss.backward()
        self.optimizerCritic.step()


        #*******************************************************************************************   Training V  *****

        v_loss, new_q_value_prediction = self.evaluate_lossV(state, actionPi, entropy, v_predicion)

        self.value.zero_grad()
        v_loss.backward()
        self.value.stepg()

        #******************************************************************************************   Training Pi  *****

        policy_loss =self.evaluate_lossPi(entropy, new_q_value_prediction)

        self.actor.zero_grad()
        policy_loss.backward()
        self.actor.stepg()


        #******************************************************************************************   Update target  ***

        # finally i update the target according to the polyak hyperparameter

        for target_parameter, parameter in zip(self.target_value.parameters(), self.value.parameters()):
            target_parameter.data.copy_(  # the underscore replace the results in data
                target_parameter.data * (self.polyak) + parameter.data * (1 -self.polyak)
            )

    #*******************************************  |save & load| ********************************************************

    def saveModel(self, name):
        os_mem.save_model(name, self.actor, self.critic_1, self.critic_2, self.value, self.target_value)

    def loadModel(self, paths):
        os_mem.load_model(paths)



