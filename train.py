from typing import List, Tuple, Dict

import tensorflow as tf
import pickle

import mapl_cirup
from learning_util import LearnTrajectoryInstance

class DDCModel(tf.keras.Model):
    def __init__(self, ddn, id2statevar, id2actvar, **kwargs):
        super().__init__(**kwargs)

        self._mc = mapl_cirup.MaplCirup(ddn, id2actvar)
        self._ddc = self._mc.get_ddc()
        self._var_num = self._mc.var_num()
        self._id2statevar = id2statevar
        self._id2actvar = id2actvar

        self.utility_param: List[tf.Variable] = self._ddc.get_param()

    def transform_x(self, x):
        init_states = []
        for init_state in x["init_states"]:
            idx  = self._mc.interface_state_to_idx(init_state)
            init_states.append(idx)
        x["init_states"] = tf.convert_to_tensor(init_states)

    def call(self, x):
        states = x['init_states']
        trajectories = x['trajectories']
        exp_rewards = []

        for el, init_state_idx in zip(trajectories, states):
            exp_rewards_traj = []
            old_interface = tf.one_hot(init_state_idx, 2 ** self._var_num, dtype=tf.float32)
            for act in el:
                new_interface, exp_reward = self._ddc.filter(old_interface, act)  # circuit evaluation
                exp_rewards_traj.append(tf.reduce_sum(exp_reward))
                old_interface = new_interface
            exp_rewards.append(exp_rewards_traj)

        return tf.convert_to_tensor(exp_rewards)

def train(ddn, x, y, id2statevar, id2actvar, lr, epochs, batch_size):
    keras_model = DDCModel(ddn, id2statevar, id2actvar)
    keras_model.transform_x(x)

    # compile sets the training parameters
    keras_model.compile(
        # By default, fit() uses tf.function(). If True the Model's logic will
        # not be wrapped in a tf.function().
        run_eagerly=True,

        # Using a built-in optimizer, configuring as an object
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),

        # Keras comes with built-in MSE error
        # However, you could use the loss function
        # defined above
        loss=tf.keras.losses.mean_squared_error,
        # loss=custom_loss
    )

    # print(f'{len(keras_model.variables)} variables: {keras_model.variables}')

    # pick_from_x = dict()
    # pick_from_x['init_states'] = x['init_states'][0:1]
    # pick_from_x['trajectories'] = x['trajectories'][0:1]

    # test_x = dict()
    # test_x['init_states'] = tf.convert_to_tensor([
    #     42
    # ])
    # test_x['trajectories'] = tf.convert_to_tensor([
    #     [[True, False, False, False], [False, True, False, False], [False, False, False, True], [False, False, False, True]]
    # ])
    # test_pred_y = keras_model(test_x)
    # test_y = tf.convert_to_tensor([
    #     [1.0, 2.0, 3.0, 4.0]
    # ])
    # test_y = tf.convert_to_tensor([
    #     1.0
    # ])
    # print(f'Test y: {test_pred_y}')

    # with tf.GradientTape() as t:
    #     # Trainable variables are automatically tracked by GradientTape
    #     test_pred_y = keras_model(test_x)
    #     print(f'Pred y: {test_pred_y}')
    #
    #     current_loss = custom_loss(test_y, test_pred_y)
    #     print(f'Loss: {current_loss}')
    #
    #     grad = t.gradient(current_loss, keras_model.variables)
    #     print(f'Grad {grad}')

    # # print(x.shape[0])
    print(keras_model.variables)
    keras_model.fit(x, y, epochs=epochs, batch_size=batch_size)
    print(keras_model.variables)

# @tf.function
# def custom_loss(y_true, y_pred):
#     error = y_true - y_pred
#     squared_error = tf.square(error)
#     return tf.reduce_mean(squared_error)


# def prepare_dataset(dataset: List[LearnTrajectoryInstance]) -> Tuple[List[Tuple[Dict[str, bool], List[Dict[str,bool]]]], List[List[int]]]:
def prepare_dataset(dataset: List[LearnTrajectoryInstance]) -> Tuple[Dict[str, tf.Tensor], tf.Tensor, Dict[int, str], Dict[int, str]]:
    """
    Split the given trajectory learning dataset into x,y with
    x a list of initial states and consecutive actions,
    and y a list with each entry a list of consecutive observed rewards.
    :param dataset:
    :return: x,y as described above
    """
    id2statevar = {idx: var for (idx, var) in enumerate(dataset[0].init_state)}
    id2actvar = {idx: var for (idx, var) in enumerate(dataset[0].actions[0])}

    action_list = []
    for el in dataset:
        trajectory = []
        for actions in el.actions:
            act_bools = [actions[id2actvar[i]] for i in range(len(id2actvar))]
            trajectory.append(act_bools)
        action_list.append(trajectory)
    tf_trajectories = tf.convert_to_tensor(action_list)

    x = {
        'init_states' : [el.init_state for el in dataset],
        'trajectories' : tf_trajectories
    }

    y = tf.convert_to_tensor([el.rewards for el in dataset], dtype=tf.float32)

    id2statevar = {a: str(b) for a,b in id2statevar.items()}
    id2actvar = {a: str(b) for a,b in id2actvar.items()}

    return x, y, id2statevar, id2actvar

def perform_learning(args):
    tf.random.set_seed(args.seed)
    with open(args.train_file, 'rb') as f:
        train_dataset = pickle.load(f)

    # prepare dataset for training
    # x = (initial state, list of actions)
    # y = list of observed rewards corresponding to the list of states
    x, y, id2statevar, id2actvar = prepare_dataset(train_dataset)
    train(args.ddn_file, x, y, id2statevar, id2actvar, args.lr, args.epochs, args.batch_size)

    #TODO: evaluation


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('ddn_file', type=str, help="Path to the DDN")
    parser.add_argument('train_file', type=str, help="Training data path")
    # parser.add_argument('valid_file', type=str, help="Validation data path")
    # parser.add_argument('model_dir', type=str, help="Model directory")
    parser.add_argument('epochs', type=int, help="Num. of training epochs")
    parser.add_argument('batch_size', type=int, help="Batch size")
    parser.add_argument('seed', type=int, help="Seed number")

    parser.add_argument('--lr', type=float, default=1,
                        help="ADAM's learning rate")

    argv = parser.parse_args()
    perform_learning(argv)
