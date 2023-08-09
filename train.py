from typing import List, Tuple, Dict

import tensorflow as tf
import pickle

from problog.logic import Term

import mapl_cirup
from learning_util import LearnTrajectoryInstance


# from typing import List

class DDCModel(tf.keras.Model):
    def __init__(self, ddn, **kwargs):
        super().__init__(**kwargs)

        self._mc = mapl_cirup.MaplCirup(ddn)
        self._ddc = self._mc.get_ddc()
        self._var_num = self._mc.var_num()

        # self.utility_param: List[tf.Variable] = self._ddc.get_param()  # TODO: Do we need this?

        # Initialize the weights to `5.0` and the bias to `0.0`
        # In practice, these should be randomly initialized
        # self.w = tf.Variable(5.0)
        # self.b = tf.Variable(0.0)

    def call(self, x):
        state, actions = x

        old_interface = tf.zeros(2 ** self._var_num, dtype=tf.float32)
        exp_rewards = tf.zeros(len(actions), dtype=tf.float32)

        # TODO: Set the initial state in the 'old_interface'

        for i, act in enumerate(actions):
            new_interface, exp_reward = self._ddc.tf_filter(old_interface, act)  # TODO: Do we need the tf version?

            exp_rewards[i] = tf.reduce_sum(exp_reward)
            old_interface = new_interface

        return exp_rewards


def train(ddn, x, y, lr, epochs, batch_size):
    keras_model = DDCModel(ddn)

    # compile sets the training parameters
    keras_model.compile(
        # By default, fit() uses tf.function().  You can
        # turn that off for debugging, but it is on now.
        # run_eagerly=False,

        # Using a built-in optimizer, configuring as an object
        optimizer=tf.keras.optimizers.SGD(learning_rate=lr),

        # Keras comes with built-in MSE error
        # However, you could use the loss function
        # defined above
        loss=tf.keras.losses.mean_squared_error,
    )

    # print(x.shape[0])
    keras_model.fit(x, y, epochs=epochs, batch_size=batch_size)


def prepare_dataset(dataset: List[LearnTrajectoryInstance]) -> Tuple[List[Tuple[Dict[Term, bool], Dict[Term,bool]]], List[List[int]]]:
    """
    Split the given trajectory learning dataset into x,y with
    x a list of initial states and consecutive actions,
    and y a list with each entry a list of consecutive observed rewards.
    :param dataset:
    :return: x,y as described above
    """
    x = [(inst.init_state, inst.actions) for inst in dataset]
    y = [inst.rewards for inst in dataset]
    return x, y


def perform_learning(args):
    with open(args.train_file, 'rb') as f:
        train_dataset = pickle.load(f)

    # prepare dataset for training
    # x = (initial state, list of actions)
    # y = list of observed rewards corresponding to the list of states
    x, y = prepare_dataset(train_dataset)
    train(args.ddn, x, y, args.lr, args.epochs, args.batch_size)

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
    # parser.add_argument('seed', type=int, help="Seed number")

    parser.add_argument('--lr', type=float, default=0.1,
                        help="SGD's learning rate")
    #
    # parser.add_argument('--weight_decay', type=float, default=defaults.weight_decay,
    #                     help="ADAM's weight decay parameter")

    argv = parser.parse_args()
    perform_learning(argv)
