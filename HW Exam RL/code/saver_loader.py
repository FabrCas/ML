import os
import torch as T
""" Module that provides save and load functions for the neural networks models"""

# Save model parameters
def save_model(env_name, actor, critic_1, critic_2, value, target_value):
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
    T.save(actor.state_dict(), actor_path)
    T.save(critic_1.state_dict(), critic1_path)
    T.save(critic_2.state_dict(), critic2_path)
    T.save(value.state_dict(), value_path)
    T.save(target_value.state_dict(), target_value_path)

    paths = [actor_path, critic1_path,critic2_path, value_path, target_value_path]
    return paths

# Load model parameters
def load_model(self, paths):

    actor_path = paths[0]
    critic1_path = paths[1]
    critic2_path = paths[2]
    value_path = paths[3]
    target_value_path = paths[4]


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
