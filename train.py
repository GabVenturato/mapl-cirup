from typing import List, Tuple, Dict

import tensorflow as tf
import pickle

from problog.logic import Term

import mapl_cirup
from learning_util import LearnTrajectoryInstance


# from typing import List

class DDCModel(tf.keras.Model):
    def __init__(self, ddn, id2statevar, id2actvar, **kwargs):
        super().__init__(**kwargs)

        self._mc = mapl_cirup.MaplCirup(ddn)
        self._ddc = self._mc.get_ddc()
        self._var_num = self._mc.var_num()
        self._id2statevar = id2statevar
        self._id2actvar = id2actvar

        # self.utility_param: List[tf.Variable] = self._ddc.get_param()  # TODO: Do we need this?

        # Initialize the weights to `5.0` and the bias to `0.0`
        # In practice, these should be randomly initialized
        # self.w = tf.Variable(5.0)
        # self.b = tf.Variable(0.0)

    def call(self, x):
        states = x['init_states']
        trajectories = x['trajectories']
        old_interface = tf.zeros(2 ** self._var_num, dtype=tf.float32)
        exp_rewards = []

        # TODO: Set the initial state in the 'old_interface'

        i = 0
        for act in trajectories:
            j = 0
            act_dict = dict()
            for val in act:
                act_dict[self._id2actvar[j]] = val
                j += 1
            new_interface, exp_reward = self._ddc.tf_filter(old_interface, act_dict)  # TODO: Do we need the tf version?

            exp_rewards.append(tf.reduce_sum(exp_reward))
            old_interface = new_interface
            i += 1

        return tf.convert_to_tensor(exp_rewards)


def train(ddn, x, y, id2statevar, id2actvar, lr, epochs, batch_size):
    keras_model = DDCModel(ddn, id2statevar, id2actvar)

    # compile sets the training parameters
    keras_model.compile(
        # By default, fit() uses tf.function().  You can
        # turn that off for debugging, but it is on now.
        run_eagerly=False,

        # Using a built-in optimizer, configuring as an object
        optimizer=tf.keras.optimizers.SGD(learning_rate=lr),

        # Keras comes with built-in MSE error
        # However, you could use the loss function
        # defined above
        loss=tf.keras.losses.mean_squared_error,
    )

    # print(x.shape[0])
    keras_model.fit(x, y, epochs=epochs, batch_size=batch_size)


# def prepare_dataset(dataset: List[LearnTrajectoryInstance]) -> Tuple[List[Tuple[Dict[str, bool], List[Dict[str,bool]]]], List[List[int]]]:
def prepare_dataset(dataset: List[LearnTrajectoryInstance]) -> Tuple[Dict[str, tf.Tensor], tf.Tensor, Dict[int, str], Dict[int, str]]:
    """
    Split the given trajectory learning dataset into x,y with
    x a list of initial states and consecutive actions,
    and y a list with each entry a list of consecutive observed rewards.
    :param dataset:
    :return: x,y as described above
    """
    # def termdict2strdict(termdict: Dict[Term, bool]) -> Dict[str, bool]:
    #     return {str(t) : v for (t, v) in termdict.items()}
    id2statevar = {idx: var for (idx, var) in enumerate(dataset[0].init_state)}
    id2actvar = {idx: var for (idx, var) in enumerate(dataset[0].actions[0])}

    state_list = []
    for el in dataset:
        state = [el.init_state[id2statevar[i]] for i in range(len(id2statevar))]
        state_list.append(state)
    tf_states = tf.convert_to_tensor(state_list)

    action_list = []
    for el in dataset:
        trajectory = []
        for actions in el.actions:
            act_bools = [actions[id2actvar[i]] for i in range(len(id2actvar))]
            trajectory.append(act_bools)
        action_list.append(trajectory)
    tf_trajectories = tf.convert_to_tensor(action_list)

    x = {
        'init_states' : tf_states,
        'trajectories' : tf_trajectories
    }

    y = tf.convert_to_tensor([el.rewards for el in dataset])

    id2statevar = {a: str(b) for a,b in id2statevar.items()}
    id2actvar = {a: str(b) for a,b in id2actvar.items()}

    return x, y, id2statevar, id2actvar

def perform_learning(args):
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
    # parser.add_argument('seed', type=int, help="Seed number")

    parser.add_argument('--lr', type=float, default=0.1,
                        help="SGD's learning rate")
    #
    # parser.add_argument('--weight_decay', type=float, default=defaults.weight_decay,
    #                     help="ADAM's weight decay parameter")

    argv = parser.parse_args()
    perform_learning(argv)
