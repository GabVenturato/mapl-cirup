# Generates synth example with a chain structure
# Outputs files '*.pl' for mapl-cirup, and '*.dat' for SPUDD.

import copy
from random import seed, random, randint

seed(1)  # seed random number generator

VAR_NAME = 's'
VAR_IDX_CHAR = '_'
DEC_NAME = 'd'


def gen_probability():
    return round(random(), 2)

def clean_var(var):
    # Remove the 'x(_)' from a variable name, e.g. "x(s1)" -> "s1"
    if var[0] == 'x':
        return var[2:-1]


def enumeration_next(state):
    next_state = copy.deepcopy(state)
    for var, val in state.items():
        if not val:
            next_state[var] = True
        else:
            next_state[var] = False
            return next_state

    return None


# P(Y_i | X_i, D_1, ..., D_n)
def gen_transition(dec_num, var_num):
    transition = dict()

    for i in range(1, var_num+1):
        distribution = []
        trans_state = dict()
        trans_state[VAR_NAME+str(i)] = True
        for j in range(1, dec_num+1):
            trans_state[DEC_NAME + str(j)] = True

        while trans_state is not None:
            distribution.append((trans_state, gen_probability()))
            trans_state = enumeration_next(trans_state)

        transition["x(" + VAR_NAME + str(i) + ")"] = distribution

    return transition


# P(Y_i | X_i-1, Y_i-1)
def gen_structure(var_num):
    structure = dict()

    if var_num >= 2:
        for i in range(2, var_num + 1):
            distribution = []
            struct_state = dict()
            struct_state[VAR_NAME + str(i - 1)] = True
            struct_state["x(" + VAR_NAME + str(i - 1) + ")"] = True

            while struct_state is not None:
                distribution.append((struct_state, gen_probability()))
                struct_state = enumeration_next(struct_state)

            structure["x(" + VAR_NAME + str(i) + ")"] = distribution

    return structure


# R(X_i = <int>)
def gen_reward(dec_num, var_num):
    reward = dict()

    for i in range(1, dec_num+1):
        reward[DEC_NAME + str(i)] = randint(-1 * dec_num, dec_num)

    for i in range(1, var_num+1):
        reward[VAR_NAME + str(i)] = randint(-2 * var_num, 2 * var_num)

    return reward


def print_rules(f, rules):
    for var, state_prob in rules.items():
        for state, prob in state_prob:
            state_list = []
            for s_name, s_val in state.items():
                state_list.append(s_name if s_val else ("\\+%s" % s_name))
            f.write("%s::%s :- %s.\n" % (prob, var, ", ".join(state_list)))
        f.write("\n")


def gen_mc_model(dec_num, var_num, transition, structure, reward):
    filename = "synth_struct_chain_d" + str(dec_num) + "_v" + str(var_num) + ".pl"
    f = open(filename, "w")

    # Decisions
    f.write("% Decisions\n")
    decisions = ["?::" + DEC_NAME + str(d) for d in range(1, dec_num+1)]
    f.write("; ".join(decisions) + ".\n\n")

    # State variables
    f.write("% State Variables\n")
    state_vars = [VAR_NAME + str(i) for i in range(1, var_num+1)]
    f.write("state_variables(%s).\n\n" % ", ".join(state_vars))

    # Transition
    f.write("% Transition\n")
    print_rules(f, transition)

    # Structure
    f.write("% Structure\n")
    print_rules(f, structure)

    # Rewards
    f.write("% Rewards\n")
    for var, val in reward.items():
        f.write("utility(%s, %s).\n" % (var, str(val)))

    f.close()


def gen_spudd_transition(var_num):
    transition = dict()

    for i in range(1, var_num + 1):
        for j in range(1, (2 ** (i-1)) + 1):
            distribution = []
            trans_state = dict()
            if i > 1:
                for h in range(1, (2 ** (i - 2)) + 1):
                    trans_state[VAR_NAME + str(i-1) + VAR_IDX_CHAR + str(h)] = True
            trans_state[VAR_NAME + str(i) + VAR_IDX_CHAR + str(j)] = True

            while trans_state is not None:
                distribution.append((trans_state, gen_probability()))
                trans_state = enumeration_next(trans_state)

            transition[VAR_NAME + str(i) + VAR_IDX_CHAR + str(j)] = distribution

    return transition


def get_var_trans_str(depth, state_vars, transition, curr_state):
    if depth == 0:
        for state, prob in transition:
            if state == curr_state:
                return "(%s %s)" % (str(prob), str(round(1-prob,2)))
    else:
        curr_var = state_vars[0]
        trans_str = "(" + curr_var + " (t "
        curr_state[curr_var] = True
        trans_str += get_var_trans_str(depth - 1, state_vars[1:], transition, curr_state)
        trans_str += ") (f "
        curr_state[curr_var] = False
        trans_str += get_var_trans_str(depth - 1, state_vars[1:], transition, curr_state)
        trans_str += "))"

        return trans_str


def gen_spudd_model(dec_num, var_num, reward):
    filename = "synth_struct_chain_d" + str(dec_num) + "_v" + str(var_num) + ".dat"
    f = open(filename, "w")

    # Variables
    variables = []
    for i in range(1, var_num + 1):
        for j in range(1, (2**(i-1))+1):
            variables.append("(%s t f)" % (VAR_NAME + str(i) + VAR_IDX_CHAR + str(j)))

    f.write("(variables " + " ".join(variables) + ")\n\n")

    # Transition
    for d in range(1, (2**dec_num)+1):
        trans = gen_spudd_transition(var_num)

        f.write("action a%s\n" % d)
        for var in trans:
            state, prob = trans[var][0]
            state_vars = list(state.keys())
            var_trans_str = get_var_trans_str(len(state_vars), state_vars, trans[var], dict())
            f.write(var + " " + var_trans_str + "\n")
        f.write("endaction\n\n")

    # Reward
    f.write("reward [+\n")
    for i in range(1, var_num + 1):
        for j in range(1, (2**(i-1))+1):
            f.write("\t(%s (t (%s)) (f (0.0)))\n" % (VAR_NAME + str(i) + VAR_IDX_CHAR + str(j), reward[VAR_NAME + str(i)]))
    f.write("]\n")

    f.write("\ndiscount 0.900000\ntolerance 0.1\n")
    f.close()


if __name__ == '__main__':
    for dec_num in range(1, 2):
        for var_num in range(11, 16):
            trans = gen_transition(dec_num, var_num)
            print("\nTransition:")
            for el in trans:
                print(str(el) + " => " + str(trans[el]))

            print("\nStructure:")
            struct = gen_structure(var_num)
            for el in struct:
                print(str(el) + " => " + str(struct[el]))

            print("\nReward:")
            rew = gen_reward(dec_num, var_num)
            for el in rew:
                print(str(el) + " => " + str(rew[el]))

            gen_mc_model(dec_num, var_num, trans, struct, rew)
            if var_num <= 5:
                gen_spudd_model(dec_num, var_num, rew)
