import os
import pickle
import sys
import random
from typing import Set, List, Dict


from problog.clausedb import ClauseDB
from problog.logic import Term, Constant, Clause, Not, And, AnnotatedDisjunction
from problog.program import PrologFile
from problog.engine import DefaultEngine
from problog.tasks.sample import sample as problog_sample

from learning_util import CompleteTrajectoryInstance, LearnTrajectoryInstance, TrajectoryInstance


def get_decision_terms(db: ClauseDB) -> List[Set[Term]]:
    """
    Retrieve decisions terms.
    The decisions in the program must be specified using (multiple) decision predicate definitions.
    The arguments of the decision atom forms one set of exclusive disjunction decisions.
    :return: A list of decision sets, where each decision set indicates an exclusive disjunction.
    """
    decisions = list()
    for entry in db:
        if type(entry) == Term and entry.functor == "decisions":
            decisions.append(set(entry.args))
    return decisions


def get_state_variables(db: ClauseDB) -> List[Term]:
    """
    Retrieve state variables.
    We expect exactly one state_variables(x1,x2,..) to be present in the program.
    :return: A list of Terms, one for each state variable atom.
    """
    for entry in db:
        if type(entry) == Term and entry.functor == "state_variables":
            return list(entry.args)
    return list()


def get_reward_dict(db: ClauseDB) -> Dict[Term, int]:
    """
    Retrieve rewards.
    :param db: Parsed program.
    :return: A dictionary {reward: val}.
    """
    reward_dict = dict(db.engine.query(db, Term("utility", None, None)))
    return {x: v.functor for (x, v) in reward_dict.items()}


def _compute_and_set_rewards(db: ClauseDB, dataset):
    reward_dict = get_reward_dict(db)
    # decisions: List[Set[Term]] = get_decision_terms(db)

    def _hardcoded_reward_func(state, actions):
        # r0 :- huc, wet.
        # r1 :- huc, \+wet.
        # r3 :- \+huc, \+wet.
        # wet, office, getu, buyc
        reward = 0
        is_r0 = state[Term("huc")] and state[Term("wet")]
        is_r1 = state[Term("huc")] and not state[Term("wet")]
        is_r3 = not state[Term("huc")] and not state[Term("wet")]
        reward += is_r0 * reward_dict[Term("r0")]
        reward += is_r1 * reward_dict[Term("r1")]
        reward += is_r3 * reward_dict[Term("r3")]
        reward += state[Term("wet")] * reward_dict[Term("wet")]
        reward += state[Term("office")] * reward_dict[Term("office")]
        reward += actions[Term("getu")] * reward_dict[Term("getu")]
        reward += actions[Term("buyc")] * reward_dict[Term("buyc")]
        return reward

    for inst in dataset:
        inst.rewards = []
        for index in range(0, len(inst.trajectory)-1):  # last state is irrelevant.
            # compute reward based on previous state (index) and its actions.
            # set previous and current states as observed state_var and state_var_next,
            transition_reward = _hardcoded_reward_func(inst.trajectory[index], inst.actions[index])
            inst.rewards.append(transition_reward)


# def _compute_and_set_rewards(db: ClauseDB, dataset):
#     # While this should work, sampling the utilities seems very slow (for some reason?), and we need another approach or figure out why it is slow.
#     # extract utility function form db
#     reward_dict = get_reward_dict(db)
#     state_vars: List[Term] = get_state_variables(db)
#     decisions: List[Set[Term]] = get_decision_terms(db)
#     decision_vars = list({decision for decision_set in decisions for decision in decision_set})
#
#     # add query utility to db2
#     db = db.extend()
#     for reward_term in reward_dict:
#         db.add_fact(Term("query", reward_term))
#
#     b2term = {True: Term("true"), False: Term("false")}
#
#     # add definition for current state and action
#     # Turn each decision into a uniform probabilistic fact
#     for decision_var in decision_vars:
#         db.add_fact(decision_var.with_probability(Constant(0.5)))
#
#     for state_var in state_vars:
#         db.add_fact(state_var.with_probability(Constant(0.5)))
#
#     # for each trajectory, compute reward
#     for inst in dataset:
#         inst.rewards = []
#         for index in range(1,len(inst.trajectory)):
#             db2 = db.extend()
#             # set previous and current states as observed state_var and state_var_next,
#             for (state_var, state_var_value) in inst.trajectory[index-1].items():
#                 db2.add_fact(Term("evidence", state_var, b2term[state_var_value]))
#             for (state_var, state_var_value) in inst.trajectory[index].items():
#                 db2.add_fact(Term("evidence", Term("x", state_var), b2term[state_var_value]))
#             for (action_var, action_var_value) in inst.actions[index-1].items():
#                 db2.add_fact(Term("evidence", action_var, b2term[action_var_value]))
#
#             # sample utility variables, these should be deterministic now.
#             utility_vars = list(problog_sample(db2, n=1, format="dict", propagate_evidence=True))[0]
#             print(utility_vars)
#             # compute total reward per utility variable and reward function
#             state_reward = 0
#             for reward_term, val in reward_dict.items():
#                 if utility_vars[reward_term]:
#                     state_reward += val
#             # update inst rewards
#             inst.rewards.append(state_reward)


def create_dataset(db: ClauseDB, nb_samples: int = 5, trajectory_length: int = 3):
    # create sampled trajectories with actions
    dataset = create_samples(db, nb_samples, trajectory_length)

    # set rewards
    _compute_and_set_rewards(db, dataset)
    dataset = [CompleteTrajectoryInstance.from_trajectory(inst) for inst in dataset]

    # for debugging
    # for inst in dataset:
    #    print(CompleteTrajectoryInstance.from_trajectory(inst))
    #    print("next instance\n")

    # remove full
    dataset = [LearnTrajectoryInstance.from_complete_trajectory(inst) for inst in dataset]
    return dataset


def create_samples(db: ClauseDB, nb_samples=5, trajectory_length=3, init_seed=None):
    # fix init_seed
    if init_seed is None:
        init_seed = random.randint(0, 20000000000)

    # get relevant atoms
    decisions: List[Set[Term]] = get_decision_terms(db)
    state_vars: List[Term] = get_state_variables(db)
    next_state_vars = [Term("x", state_var) for state_var in state_vars]

    db2 = db.extend()
    # Turn each decision into a uniform probabilistic fact
    for decision_set in decisions:
        if len(decision_set) == 1:
            decision = decision_set.pop()
            db2.add_fact(decision.with_probability(Constant(0.5)))
        else:
            # AD
            nb_vars = len(decision_set)
            prob_per_var = Constant(1.0 / nb_vars)
            ad_heads = list({d.with_probability(prob_per_var) for d in decision_set})
            db2.add_clause(AnnotatedDisjunction(heads=ad_heads, body=Term("true")))

    # Turn each state variable into a uniform probabilistic fact.
    for state_var in state_vars:
        db2.add_fact(state_var.with_probability(Constant(0.5)))

    # Add query for all atoms.
    decision_vars = list({decision for decision_set in decisions for decision in decision_set})
    for decision_var in decision_vars:
        db2.add_fact(Term('query', decision_var))
    for state_var in state_vars:
        db2.add_fact(Term('query', state_var))
    for next_state_var in next_state_vars:
        db2.add_fact(Term('query', next_state_var))

    # create samples
    dataset = []
    curr_seed = init_seed
    for i in range(nb_samples):
        curr_seed += 1
        random.seed(a=curr_seed)
        trajectory, actions = _create_samples_trajectory_aux(
            db=db2,
            state_vars=state_vars,
            decision_vars=decision_vars,
            next_state_vars=next_state_vars,
            trajectory_length=trajectory_length)
        inst = TrajectoryInstance(trajectory, actions, rewards=[], seed=curr_seed)
        dataset.append(inst)
    return dataset


def _create_samples_trajectory_aux(
        db: ClauseDB,
        state_vars: List[Term],
        decision_vars: List[Term],
        next_state_vars: List[Term],
        trajectory_length=3):
    """
    Sample a trajectory
    :param db:
    :param trajectory_length: The number of states to sample. At least 2.
    :param state_vars: The list of state variables present in db.
    :param decision_vars: The list of decision vars present in db.
    :param next_state_vars: The list of next state vars present in db.
    :return: A list of states, and one of actions, each a dict of Terms to bool.
    For example,
    [{raining: False, office: True},...,{raining:True, office:False} ],
    [{move: True, buyc: False},...{move:False, buyc:True}]
    The size of the state list is equal to trajectory_length list.
    The size of the action list is one less.
    """
    assert trajectory_length > 1
    trajectory = []
    actions = []

    # sample from program to get state 0 and state 1
    curr_sample = list(problog_sample(db, n=1, format="dict"))[0]
    state0 = {state_var: curr_sample[state_var] for state_var in state_vars}
    decisions0 = {decision: curr_sample[decision] for decision in decision_vars}
    state1 = {next_state_var.args[0]: curr_sample[next_state_var] for next_state_var in
              next_state_vars}
    trajectory.append(state0)
    actions.append(decisions0)
    trajectory.append(state1)

    for i in range(trajectory_length - 2):
        # add previously determined next state as current state
        db2 = db.extend()
        for next_state_var in next_state_vars:
            state_var_value = Term("true") if curr_sample[next_state_var] else Term("false")
            state_var = next_state_var.args[0]
            db2.add_fact(Term("evidence", state_var, state_var_value))
        # - sample again
        curr_sample = list(problog_sample(db, n=1, format="dict"))[0]
        next_decision = {decision: curr_sample[decision] for decision in decision_vars}
        next_state = {next_state_var.args[0]: curr_sample[next_state_var] for next_state_var in
                      next_state_vars}
        actions.append(next_decision)
        trajectory.append(next_state)
    return trajectory, actions


def _generate_trajectories(db: ClauseDB,
                           nb_samples: int,
                           traj_length: int):
    """

    :param db: A ProbLog program containing utility statements with true utilities.
    It must not contain any decision statements.
    It must contain state_variables(wet, raining,...). and decisions(move, delc, ...).
    :param nb_samples: number of trajectory samples to generate
    :param traj_length: Trajectory length, counting the initial and last state.
    :return:
    """
    # get sample trajectories
    # db can't have decisions yet, because does not know how to sample.
    dataset = create_dataset(db=db, nb_samples=nb_samples, trajectory_length=traj_length)
    return dataset


def _add_reward_function(db: ClauseDB, reward_dict: Dict[Term, float]):
    """ Add the given reward function to the given db. """
    for reward_term, reward_value in reward_dict.items():
        db.add_fact(Term("utility", reward_term, Constant(reward_value)))


def get_random_reward_dict(low_val=-10, high_val=10) -> Dict[Term, float]:
    """ Get random reward dict with each value in [low_val, high_val]. """
    # WARNING: the reward function's structure is hardcoded in _hardcoded_reward_func
    # do not change this set below without considering other code!
    reward_terms = {
        Term("r0"), Term("r1"), Term("r3"),
        Term("wet"), Term("office"),
        Term("getu"), Term("buyc"),
    }
    # to avoid reward 0, we create a sequence of all but 0
    reward_set = list(range(low_val, high_val, 1))
    reward_set.remove(0)
    rewards_sampled = random.choices(reward_set, k=len(reward_terms))
    reward_dict = {r_term: rewards_sampled[idx] for (idx, r_term) in enumerate(reward_terms)}
    return reward_dict


def _add_decisions(db: ClauseDB):
    """
    Add decisions to db.
    Uses decisions(move, delc, ...) present in the db to determine the decisions.
    :param db:
    """
    decisions: List[Set[Term]] = get_decision_terms(db)
    dec_prob = Term("?")
    for decision_set in decisions:
        if len(decision_set) == 1:
            decision = decision_set.pop()
            db.add_fact(decision.with_probability(dec_prob))
        else:
            # AD
            AD_heads = list({d.with_probability(dec_prob) for d in decision_set})
            AD = AnnotatedDisjunction(heads=AD_heads, body=Term("true"))
            db.add_clause(AD)


def main(argv):
    print(argv)
    print("Warning, hardcoded reward function.")  # cf. _hardcoded_reward_func
    args = argparser().parse_args(argv)

    # settings
    # settings - action
    action_create_dataset = args.create_dataset
    action_create_true_model = args.create_true_model
    action_create_init_model = args.create_init_model
    if not any([action_create_dataset, action_create_true_model, action_create_init_model]):
        print("No generate action was selected! Use --help to to see what args to use.")
        return

    # settings - filepaths
    filepath_blank_model = args.blank_input_filepath  # no decisions, no utilities
    if filepath_blank_model is None and (action_create_true_model or action_create_init_model):
        print("blank_input_filepath is required when creating a true or initial model.",
              file=sys.stderr)
        return
    filepath_true_model = args.true_filepath  # complete, true
    if filepath_true_model is None and action_create_true_model:
        print("true_filepath is required when creating a true model.",
              file=sys.stderr)
        return
    filepath_true_wo_dec = args.true_wo_dec_filepath  # true utility but no decisions
    if filepath_true_wo_dec is None and (action_create_true_model or action_create_dataset):
        print("true_wo_dec_filepath is required when creating a true model or dataset.",
              file=sys.stderr)
        return
    filepath_init_model = args.init_filepath
    if filepath_init_model is None and action_create_init_model:
        print("init_filepath is required when creating an initial model",
              file=sys.stderr)
        return
    filepath_traj_folder = args.dataset_folder

    # settings - parameters
    n = args.dataset_size
    traj_length = args.traj_length
    seed_dataset = args.seed_traj
    seed_true = args.seed_true
    seed_init = args.seed_init
    database_filename = f"dataset_n{n}_trajlen{traj_length}_seed{seed_dataset}.pickle"
    filepath_traj_database = os.path.join(filepath_traj_folder, database_filename)

    # create true model without decisions
    # i.e., fill in a reward function
    if action_create_true_model:
        print("Generating true .pl files (chooses reward function).")
        random.seed(a=seed_true)
        db = DefaultEngine().prepare(PrologFile(filepath_blank_model))
        db_true_wo_dec = db.extend()
        reward_dict_true = get_random_reward_dict()
        _add_reward_function(db_true_wo_dec, reward_dict_true)
        with open(filepath_true_wo_dec, "w") as f:
            f.write(db_true_wo_dec.to_prolog())

        db_true = db_true_wo_dec.extend()
        _add_decisions(db_true)
        with open(filepath_true_model, "w") as f:
            f.write(db_true.to_prolog())
        print("\tdone.")
        del db_true
        del db_true_wo_dec

    # sample trajectories with rewards
    # db can't have decisions yet, because does not know how to sample.
    # it should have the true reward dictionary already though.
    if action_create_dataset:
        print(f"Generating dataset of {n} trajectories, of size {traj_length}.")
        random.seed(a=seed_dataset)
        db_true_wo_dec = DefaultEngine().prepare(PrologFile(filepath_true_wo_dec))
        dataset = _generate_trajectories(db=db_true_wo_dec, nb_samples=n, traj_length=traj_length)
        with open(filepath_traj_database, "wb") as f:
            pickle.dump(dataset, f)
        print(f"\tdone.")
        # for inst in dataset:
        #     print(inst)
        del db_true_wo_dec
        del dataset

    # create initial learning template
    if action_create_init_model:
        print(f"Generating initial learning .pl files (chooses random reward function).")
        random.seed(a=seed_init)
        db = DefaultEngine().prepare(PrologFile(filepath_blank_model))
        reward_dict_true = get_random_reward_dict()
        _add_reward_function(db, reward_dict_true)
        _add_decisions(db)
        with open(filepath_init_model, "w") as f:
            f.write(db.to_prolog())
        print("\tdone.")
        del db


def argparser():
    import argparse

    parser = argparse.ArgumentParser(description="Create dataset for learning")
    # dataset
    parser.add_argument('--dataset_size', type=int, required=False,
                        help="Size of the dataset", default=10)
    parser.add_argument('--traj_length', type=int, required=False,
                        help="Length of each trajectory in the dataset", default=5)
    parser.add_argument('-s', '--seed_traj',
                        help='seed used to generate trajectories',
                        type=int, default=1000)
    parser.add_argument('-d', "--create_dataset", action='store_true',
                        help="indicates that the trajectory dataset must be created. ")

    # generate true file
    parser.add_argument('--seed_true',
                        help='seed used to generate random true utility function',
                        type=int, default=27854867)
    parser.add_argument('-t', "--create_true_model", action='store_true',
                        help="indicates that a true model must be created. ")

    # generate init file
    parser.add_argument('--seed_init',
                        help='seed used to generate random initial utility functions',
                        type=int, default=1757887)
    parser.add_argument('-i', "--create_init_model", action='store_true',
                        help="indicates that an initial model must be created. ")

    # filepaths
    parser.add_argument('--blank_input_filepath', required=False,
                        help="Filepath to .pl containing the coffee example without "
                             "utility, nor decisions."
                             "REQUIRED when creating: true model, or initial model")
    parser.add_argument('--true_filepath', required=False,
                        help="Filepath to write the true .pl model to. "
                             "REQUIRED when creating: true model")
    parser.add_argument('--true_wo_dec_filepath', required=False,
                        help="Filepath of the true .pl model but without decisions. "
                             "REQUIRED when creating: true_model, or dataset")
    parser.add_argument('--init_filepath', required=False,
                        help="Filepath to write the initial .pl model to. "
                             "REQUIRED when creating: initial model")
    parser.add_argument('--dataset_folder', required=False,
                        help="Filepath of the folder where to write the dataset to. ",
                        default="./")
    # parser.add_argument('--dataset_filepath', required=False,
    #                     help="Filepath of write the dataset to. "
    #                          "REQUIRED when creating: dataset")
    return parser


"""
Assumes ground programs
Assumes state variables are stored as a statement
        state_variables(a,b,c).
    , and that those atoms are not already defined.
Assumes decision variables are stored as a statement
        decisions(a,b,c). % AD a ; b ; c
        decisions(d,e,f). % AD d ; e ; f
    , and that those atoms are not already defined.
Assumes next state variables are defined using an x predicate symbol, e.g., x(a), x(b), etc.
"""

if __name__ == '__main__':
    main(sys.argv[1:])
