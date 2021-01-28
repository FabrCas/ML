from criticNet import CriticNetwork
from actorNet import ActorNetwork
from valueNet import ValueNetwork
from replayBuffer import ReplayBuffer
import torch as T
from copy import deepcopy
import torch.nn.functional as F

class SoftActorCriticAgent():
    def __init__(self, alpha = 0.2, input_dims = None, env = None, gamma= 0.99, n_actions = None,
                 max_size= 100000, tau = 0.005, layer_size = 256, layer2_size=256, batch_size= 32, reward_scale= 2, polyak=0.995):
        self.gamma= gamma
        self.tau = tau
        self.alpha = alpha   # Definition of the temperature parameter
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size =  batch_size
        self.n_actions = n_actions
        self.polyak = polyak


        """ 2 critics, minimum od the evalution of this state for these 2 networks in the calculation of the loss function
         by the critic part"""

        self.actor = ActorNetwork(input_dims, n_actions=n_actions,
                    name='actor', max_action=env.action_space.high)
        self.critic_1 = CriticNetwork(input_dims, n_actions=n_actions,
                    name='critic_1')
        self.critic_2 = CriticNetwork(input_dims, n_actions=n_actions,
                    name='critic_2')

        # 2 ways of acting for estimating the value function:
        # 1) define and learn a specific neural network
        # 2) evaluate the value of a certain state V(st) from the expected values of the difference between
        # the q function Q(st, at) minus the entropy log(at|st), with at in reference to a certain policy pi

        #self.target_value_1= ValueNetwork(beta, input_dims, name='target_value_1'))
        #self.target_value_2 = ValueNetwork(beta, input_dims, name='target_value_2')
        # we use the second choice
        self.targetQ1 = deepcopy(self.critic_1)
        self.targetQ2 = deepcopy(self.critic_2)
        # so we have to freeze the target, hence V(st) from the optimizer, it will be update manually
        self.targetQ1.freezeParameters()
        self.targetQ2.freezeParameters()

        self.scale = reward_scale

        """ this function set parameters of target value network equals to  parameters of the target value network to start"""
        #self.update_network_parameters(tau=1)

    """
    def choose_action(self, observation):
        state = T.Tensor([observation]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions.cpu().detach().numpy()[0]"""

    """
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)"""

    """
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value_1.named_parameters()
        value_params = self.target_value_2.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau * value_state_dict[name].clone() + \
                                     (1 - tau) * target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)
    """
    """
    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint() """
    """
    def learn(self):
        if self.memory.pointer < self.batch_size:
            return

        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)

        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0

        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale * reward + self.gamma * value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()"""
# **********************************************************************************************************************
    # optimizers handles in the nns

    # params -> data from experience replay
    def loss_evaluation_Q(self,data): #state, action, reward, new_State, done):
        # compute the q value of the observation and the action according to 2 q functions
        state = data["states"]
        action = data["actions"]
        reward = data["rewards"]
        new_State = data["states_"]
        done = data["dones"]
        # this part can be optimized with stochastic gradients
        q1 = self.critic_1.forward(state, action)
        q2 = self.critic_2.forward(state, action)

        # we want gradient out of the target
        with T.no_grad():
            action2, log_prob= self.actor.sample_normal(new_State)

            # q values from the target functions, from the action chosen by the policy, in the new state
            q1_target = self.targetQ1.forward(new_State, action2)
            q2_target = self.targetQ2.forward(new_State, action2)
            # we take the minimum between these 2 q target values
            q_target = T.min(q1_target,q2_target)
            # with (q_target - self.alpha * log_prob) = V(st)
            target = reward + self.gamma * (1 - done) * (q_target - self.alpha * log_prob)

        #evaluation of the mean square error loss
        lq1 = ((q1 - target)**2).mean()
        lq2 = ((q2 - target)**2).mean()
        loss = lq1 + lq2
        return loss

    # compution for the loss of the policy
    def loss_evaluation_PI(self, data): #state):
        state = data['states']

        action, log_prob = self.actor.sample_normal(state)   #1.12.05
        # i need to compute q1 and q2 of pi, so take the action from pi and use the critic networks, then take the min
        q1_pi= self.critic_1.forward(state, action)
        q2_pi = self.critic_1.forward(state, action)
        q_pi = T.min(q1_pi, q2_pi)

        # evaluating the loss of the policy
        lossPi= (self.alpha * log_prob - q_pi).mean() #formula 6, alpha has been omitted

        return lossPi

    def choose_action(self, state):
        state = T.Tensor([state])
        with T.no_grad():
            actions, _ = self.actor.sample_normal(state)#T.as_tensor(state, dtype=T.float32))
            actions = actions.squeeze()
            print(actions)
            return  actions.numpy()  #transform from tensor to array and return               #actions.cpu().detach().numpy()[0]

    def save(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def sample_buffer_replay(self, batch_size):
        return self.memory.sample_buffer(batch_size)

    def learn(self,data): # states, actions, rewards, new_States, dones):
        # learn part for critic

        # gradient descent for pi
        self.actor.zero_grad()
        loss = self.loss_evaluation_PI(data)  # states)
        loss.backward()
        self.actor.stepg()


        # initialize Q optimizers for having zero gradient
        self.critic_1.zero_grad()
        self.critic_2.zero_grad()
        loss = self.loss_evaluation_Q(data) #states,actions,rewards,new_States,dones)
        loss.backward()  # compute the gradient
        self.critic_1.stepg() #apply the gradient through the optimizer
        self.critic_2.stepg()

        # learn part for actor

        # now the same for the policy
        # we have to block the gradient for Q parameters
        # self.critic_1.freezeParameters()
        # self.critic_2.freezeParameters()

        # re-active gradient parameters
        # self.critic_1.unfreezeParameters()
        # self.critic_2.unfreezeParameters()

        # update target networks off the gradient
        with T.no_grad():
            for params, targParams1 in zip(self.critic_1.parameters(), self.targetQ1.parameters()):
                targParams1.data.mul_(self.polyak)  #the underscore replace the results of multiplication in data
                targParams1.data.add_((1- self.polyak) * params.data)

            for params, targParams2 in zip(self.critic_2.parameters(), self.targetQ2.parameters()):
                targParams2.data.mul_(self.polyak)  #the underscore replace the results of multiplication in data
                targParams2.data.add_((1- self.polyak) * params.data)





