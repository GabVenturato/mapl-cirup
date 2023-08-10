import pickle
from typing import Dict, List

import numpy as np
from problog.clausedb import ClauseDB

from problog.logic import Term
from problog.program import PrologFile
from problog.engine import DefaultEngine

import matplotlib.pyplot as plt

def get_reward_params_from_db(db: ClauseDB) -> Dict[Term, int]:
    """
    Retrieve rewards.
    :param db: Parsed program.
    :return: A dictionary {reward: val}.
    """
    reward_dict = dict(db.engine.query(db, Term("utility", None, None)))
    return {x: v.functor for (x, v) in reward_dict.items()}


def print_evaluation(learned_params_dict: [Term, float],
                     true_params_dict: Dict[Term, float]):
    assert len(learned_params_dict) == len(true_params_dict)
    print("-- Printing parameters --")
    print("name -- true val -- learned val (abs error  | rel err %)")
    abs_errors = []
    rel_errors = []
    for pname in learned_params_dict:
        learned_val = learned_params_dict[pname]
        true_val = true_params_dict[pname]
        abs_error = abs(learned_val - true_val)
        rel_error = abs_error / true_val if true_val != 0 else "nan"
        print(f"{pname}\t{true_val}\t{learned_val}\t ({abs_error}|{rel_error}%)")
        abs_errors.append(abs_error)
        rel_errors.append(rel_error)
    abs_errors = np.array(abs_errors)
    rel_errors = np.array(rel_errors)
    print("----------")
    print(f"Average rel error {np.average(rel_errors)} (+- variance {np.var(rel_errors)})")


def create_param_dict(params: List[float], params_names: List[Term]):
    """ Create a dictionary of Term to float, i.e., param name to its reward. """
    return {pname: p for (pname, p) in zip(params_names, params)}


def get_rel_errors(learned_values, true_values: List[float]) -> List[List[float]]:
    """
    Get relative errors of the given learned values and true values.
    :param learned_values: A 2D list of learned values. Might also work with a 1D array.
    :param true_values: A 1D list of true values, with the ordering of each parameter matching
    the ordering in learned_values.
    :return: A 2D list of learned values where each value is now the relative error with respect
    to the true parameter value in the same index.
    """
    learned_values = np.array(learned_values)
    true_values = np.array(true_values)
    error_values = learned_values - true_values
    return error_values / true_values


def param_dict_to_list(param_dict: Dict[Term, float], new_order: List[Term]):
    """ transform param_dict into a list of values, ordered by the names given in new_order. """
    return [param_dict[pname] for pname in new_order]


def plot_loss_and_rel_error(loss, avg_rel_errors):
    assert len(loss) == len(avg_rel_errors)
    iterations = range(len(loss))

    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(iterations, loss, color=color)
    # ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Average relative errors (%', color=color)
    ax2.plot(iterations, avg_rel_errors, color=color, linestyle='dashed')
    # ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.savefig("loss_avg_rel_error.pdf", format="pdf", bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    # extract results
    with open("examples_learning/log.pickle", 'rb') as f:
        results = pickle.load(f)
        learned_params, losses, learned_param_names = results
    # print(losses[-1])  # the last loss is -1 because I haven't computed yet the true one, you shouldn't need it
    learned_dict = create_param_dict(learned_params[-1], learned_param_names)

    # extract true parameters
    true_filepath = "examples_learning/coffee_v6_learn_true.pl"
    model = PrologFile(true_filepath)
    db = DefaultEngine().prepare(model)
    true_dict = get_reward_params_from_db(db)

    # evaluation
    print_evaluation(learned_dict, true_dict)