import pickle
from typing import Dict, List, Set

import numpy as np
from problog.clausedb import ClauseDB

from problog.logic import Term
from problog.program import PrologFile
from problog.engine import DefaultEngine

import matplotlib.pyplot as plt

from examples_learning import create_learn_dataset


def get_reward_params_from_db(db: ClauseDB) -> Dict[Term, int]:
    """
    Retrieve rewards.
    :param db: Parsed program.
    :return: A dictionary {reward: val}.
    """
    reward_dict = dict(db.engine.query(db, Term("utility", None, None)))
    return {x: v.functor for (x, v) in reward_dict.items()}


def print_evaluation(learned_params_dict: Dict[str, float],
                     true_params_dict: Dict[str, float]):
    assert len(learned_params_dict) == len(true_params_dict)
    print("-- Printing parameters --")
    print("name".ljust(10, ' ') + "\t|\ttrue val\t|\tlearned val\t|\tabs error\t|\trel err %")
    abs_errors = []
    rel_errors = []
    for pname in true_params_dict:
        learned_val = learned_params_dict[pname]
        true_val = true_params_dict[pname]
        abs_error = abs(learned_val - true_val)
        rel_error = abs_error / abs(true_val) * 100. if true_val != 0 else "nan"
        print(f"{pname.ljust(10, ' ')}\t{true_val:10d}\t\t{learned_val:.4f}\t\t{abs_error:.4f}\t\t{rel_error:.2f}")
        abs_errors.append(abs_error)
        rel_errors.append(rel_error)
    abs_errors = np.array(abs_errors)
    rel_errors = np.array(rel_errors)
    print("----------")
    print(f"Average rel error {np.average(rel_errors):.2f}% (+- variance {np.std(rel_errors):.2f}%)")


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
    return abs(error_values) / abs(true_values)


def param_dict_to_list(param_dict: Dict[str, float], new_order: List[str]):
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
    ax2.set_ylabel('Average relative errors', color=color)
    ax2.plot(iterations, avg_rel_errors, color=color, linestyle='dashed')
    # ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.savefig("loss_avg_rel_error.pdf", format="pdf", bbox_inches='tight')
    plt.show()


def evaluate_states(db: ClauseDB,
                    learned_params: List[List[float]],
                    learned_param_names: List[str]):
    decision_vars = set()
    decision_vars = decision_vars.union(*create_learn_dataset.get_decision_terms(db))
    decision_vars = [str(x) for x in decision_vars]
    state_vars = [str(s) for s in create_learn_dataset.get_state_variables(db)]  # TODO push outside
    learned_params = np.array(learned_params)
    reward_dict = {pname: learned_params[:,i] for i, pname in enumerate(learned_param_names)}

    def _hardcoded_reward_func(_state, _action):
        # r0 :- huc, wet.
        # r1 :- huc, \+wet.
        # r3 :- \+huc, \+wet.
        # wet, office, getu, buyc
        _reward = 0
        is_r0 = "huc" in _state and "wet" in _state
        is_r1 = "huc" in _state and "wet" not in _state
        is_r3 = "huc" not in _state and "wet" not in _state
        _reward += reward_dict["r0"] * is_r0
        _reward += reward_dict["r1"] * is_r1
        _reward += reward_dict["r3"] * is_r3
        _reward += reward_dict["wet"] * ("wet" in _state)
        _reward += reward_dict["office"] * ("office" in _state)
        _reward += reward_dict["getu"] * ("getu" in _action)
        _reward += reward_dict["buyc"] * ("buyc" in _action)
        return _reward

    def _idx_to_state(_idx, _state_vars):
        # bit magic, each bit in _idx represents a state_var
        # so we check the bit using a mask to see if the state_var is true or false
        _state_list = []
        for j, svar in enumerate(_state_vars):
            mask = 1 << j
            if (_idx & mask) > 0:
                _state_list.append(svar)
            else:
                _state_list.append(f"\\+{svar}")
        return _state_list

    # go over each state, action set
    # since each decision is an AD, it is decision_vars * 2**|state_vars|
    state_rewards = []
    for decision_var in decision_vars:
        for i in range(2**len(state_vars)):
            state, action = _idx_to_state(i, state_vars), {decision_var}
            state = set(state)
            # compute its utility, add it to a list
            reward = _hardcoded_reward_func(state, action)
            state_rewards.append(reward)
    return state_rewards

def main():
    # extract results
    with open("examples_learning/log_30epochs_dataset_n100_trajlen5_seed42.pickle", 'rb') as f:
        results = pickle.load(f)
        learned_params, losses, learned_param_names = results
    # print(losses[-1])  # the last loss is -1 because I haven't computed yet the true one, you shouldn't need it
    learned_dict = create_param_dict(learned_params[-1], learned_param_names)

    # extract true parameters
    true_filepath = "examples_learning/coffee_v6_learn_true.pl"
    model = PrologFile(true_filepath)
    db = DefaultEngine().prepare(model)
    true_dict = get_reward_params_from_db(db)
    true_dict = {str(p): v for (p,v) in true_dict.items()}

    # evaluation
    print_evaluation(learned_dict, true_dict)

    # plots
    true_list = param_dict_to_list(param_dict=true_dict, new_order=learned_param_names)
    rel_errors = get_rel_errors(learned_params, true_list)
    avg_rel_errors = np.average(rel_errors, axis=1)
    plot_loss_and_rel_error(losses[:-1], avg_rel_errors[:-1])

    # utility_per_state = evaluate_states(db, learned_params[:-1], learned_param_names)
    # print("test")


if __name__ == '__main__':
    main()