"""
MArkov PLanning with CIRcuit bellman UPdates (mapl-cirup)
"""
import copy
import time
import numpy as np
import tensorflow as tf

from typing import Dict, List

from problog import get_evaluatable
from problog.clausedb import ClauseDB
from problog.formula import LogicFormula
from problog.logic import Term, Constant, Clause, Not, And
from problog.program import PrologFile
from problog.engine import DefaultEngine
from problog.sdd_formula_explicit import SDDExplicit, x_constrained_named

from ddc import DDC


class MaplCirup:
    """
    Class for mapl-cirup.

    The circuit is symbolically representing both the utility function U(s) and the policy π(s), for each explicit
    state s.
    """
    _engine = DefaultEngine(label_all=True, keep_order=True)    # Grounding engine
    _next_state_functor = 'x'                                   # Name of the functor used to indicate the next state
    _true_term = Term('true')                                   # True term used for querying
    _decision_term = Term("?")                                  # Decision term used for querying
    _horizon = 0                                                # Default future lookahead
    _discount = 1                                               # Default discount factor
    _error = 0.01                                               # Default maximum error allowed for value iteration
    _utilities: Dict[Term, List[Term]] = {}                     # Added (expected) utility parameters with their state
    _iterations_count: int = 0                                  # Number of iterations for policy convergence
    _vi_time = None                                             # Time required for value iteration
    _minimisation_on = False                                    # Activate the SDD minimisation
    _minimize_time = 0                                          # Default minimisation time

    def __init__(self, filename, minimisation=False):
        """
        DDC initialization. The overall steps are the following:
            1. Parse the two input model
            3. Fill in some class parameters to perform inference later
            4. Grounding
            5. Knowledge compilation (KC)
            6. Initialize the maximum expected utility semiring
        Notice that some operations must be in this order because they may depend on something retrieved earlier.
        :param filename: Input file in the ProbLog format
        """
        prog = self._parsing(filename)
        self._rewards = self._get_reward_func(prog)
        self._decisions = self._get_decisions(prog)
        self._state_vars = MaplCirup._get_state_vars(prog)
        self._add_state_priors(prog)
        self._add_utility_parameters(prog)
        grounded_prog = self._grounding(prog)

        print("Compiling...")
        starttime_compilation = time.time()
        sdd = self._compilation(grounded_prog)
        # TODO Print the size of the SDD after compiling (also after minimising maybe)

        if minimisation or self._minimisation_on:
            print("Minimizing...")
            starttime_minimization = time.time()
            self._minimize(sdd)
            endtime_minimization = time.time()
            self._minimize_time = endtime_minimization - starttime_minimization

        self._ddc: DDC = DDC.create_from(sdd, self._state_vars, self._rewards, self._discount)
        endtime_compilation = time.time()
        self._compile_time = endtime_compilation - starttime_compilation
        print("Compilation done! (circuit size: %s)" % self.size())

        self._remove_impossible_states()
        # (p, eu, dec) = self._ddc.maxeu()  # {'hit': False}
        # print("DDC maxeu eval: %s, %s, %s" % (p, eu, dec))
        self._ddc.print_info()

        return

    def _remove_impossible_states(self):
        imp_util = self._ddc.impossible_utilities()
        print("\nImpossible states (%s/%s):" % (len(imp_util), 2 ** len(self._state_vars)))
        to_remove = []
        for u, state in self._utilities.items():
            if str(u) in imp_util:
                print(str(state))
                to_remove.append(u)

        for u in to_remove:
            self._utilities.pop(u)

        print()

    def _parsing(self, file_path="") -> ClauseDB:
        """
        Parse the input model located at 'file_path'.
        :return: The parsed program.
        """
        return self._engine.prepare(PrologFile(file_path))

    def _get_reward_func(self, program: ClauseDB) -> Dict[Term, Constant]:
        """
        Retrieve utilities from the parsed program.
        :param program: Parsed program.
        :return: A dictionary {reward: val}.
        """
        return dict(self._engine.query(program, Term('utility', None, None)))

    def _get_decisions(self, program: ClauseDB) -> List[Term]:
        """
        Retrieve decisions from the parsed program.
        :param program: Parsed program.
        :return: A list containing all the decisions in the model.
        """
        decisions = set()

        # Retrieve the decisions
        for _, node in program.enum_nodes():
            if not node:
                continue
            node_type = type(node).__name__
            if hasattr(node, 'probability'):
                if node.probability == self._decision_term:
                    if node_type == 'choice':  # unpack from the choice node
                        decisions.add(node.functor.args[2])
                    else:
                        decisions.add(Term(node.functor, *node.args))

        return list(decisions)

    @staticmethod
    def _get_state_vars(program: ClauseDB) -> List[Term]:
        """
        Retrieve the variables representing the state. Notice that it returns only the state variables and not the
        decision ones.
        :param program: Parsed program.
        :return: A set of terms, each is a state variable.
        """
        for rule in program:
            if type(rule) == Term and rule.functor == 'state_variables':
                return list(rule.args)

        return []

    def _add_state_priors(self, parsed_prog: ClauseDB) -> None:
        for var in self._state_vars:
            new_var = copy.deepcopy(var)
            if var.probability is None:
                new_var.probability = Constant(1.0)
            parsed_prog.add_fact(new_var)

    def _add_utility_parameters(self, parsed_prog: ClauseDB) -> None:
        """
        Add parameters representing the future expected utility. They must be connected to primed variables, i.e. the
        next state in the transition function.
        TODO: Optionally add a utility parameter if the corresponding state has probability > 0.
        :param parsed_prog: Parsed program.
        :return: Void.
        """
        utility_idx: int = 0
        state = self._state_vars

        while state:
            utility_term = Term('u' + str(utility_idx))
            self._utilities[utility_term] = state
            parsed_prog.add_clause(Clause(utility_term, MaplCirup.big_and(self._wrap_in_next_state_functor(state))))
            utility_idx += 1
            state = MaplCirup.enumeration_next(state)

    def _wrap_in_next_state_functor(self, state: List[Term]) -> List[Term]:
        wrapped_state = []

        for term in state:
            if isinstance(term, Not):
                wrapped_state.append(Term(self._next_state_functor, term.args[0]).__invert__())
            else:
                wrapped_state.append(Term(self._next_state_functor, term))

        return wrapped_state

    def _grounding(self, parsed_prog: ClauseDB) -> LogicFormula:
        """
        Ground the parsed programs.
        :param parsed_prog: Parsed program.
        :return: Grounded program.
        """
        queries = self._decisions + list(self._rewards) + list(self._utilities.keys())
        queries += list(map(lambda v: Term(self._next_state_functor, v), self._state_vars))

        # fix an order to have the same circuit size at each execution
        # (for some reason the reverse order leads to smaller circuits)
        queries.sort(key=repr, reverse=True)

        queries.append(self._true_term)

        return self._engine.ground_all(parsed_prog, queries=queries)

    def _compilation(self, grounded_prog: LogicFormula) -> SDDExplicit:
        """
        Knowledge compilation into X-constrained SDDs of the model.
        :param grounded_prog: Grounded model.
        :return: The circuit for the given model.
        """
        # print("Start compilation")
        kc_class = get_evaluatable(name='sddx')
        constraints = x_constrained_named(X_named=self._decisions)
        # starttime_compilation = time.time()
        circuit: SDDExplicit = kc_class.create_from(grounded_prog, var_constraint=constraints)
        # endtime_compilation = time.time()
        # compile_time = endtime_compilation - starttime_compilation
        # print("Compilation took %s seconds." % compile_time)

        return circuit

    @staticmethod
    def _minimize(sdd: SDDExplicit) -> None:
        """
        SDD Minimization (Sec 5.5 of the advanced manual).
        """
        # If one wants to limit times more strictly. Default parameters should be: 180, 60, 30, 10.
        # self._circuit.get_manager().get_manager().set_vtree_search_time_limit(60)
        # self._circuit.get_manager().get_manager().set_vtree_fragment_time_limit(20)
        # self._circuit.get_manager().get_manager().set_vtree_operation_time_limit(10)
        # self._circuit.get_manager().get_manager().set_vtree_apply_time_limit(5)

        # The following call to 'ref()' is required otherwise the minimization removes necessary nodes
        sdd.get_root_inode().ref()
        sdd.get_manager().get_manager().minimize_limited()

    @staticmethod
    def enumeration_next(state: List[Term]) -> List[Term]:
        """
        Takes in input a state (as a list of terms), and return the next in the enumeration order.
        Assumptions:
        - all terms are binary variables
        - the first state is when all terms are true
        - the last state is when all terms are false
        Enumeration order example: [x1,x2], [¬x1,x2], [x1,¬x2], [¬x1,¬x2].
        :param state: Current state, represented as a list or Terms.
        :return: The next state in the enumeration order. Returns an empty list when the final state is given in input.
        """
        next_state: List[Term] = copy.deepcopy(state)
        for idx, term in enumerate(state):
            if isinstance(term, Not):
                next_state[idx] = term.args[0]
            else:
                next_state[idx] = term.__invert__()
                return next_state

        return []

    @staticmethod
    def big_and(terms: List[Term]) -> Term:
        """
        Transform a list of terms into a concatenation of logical ands. For example, [x,y,z] -> And(x,And(y,z)).
        :param terms: List of terms to be concatenated.
        :return: The big and concatenation. If only one term is in the list, it returns the term itself.
        """
        if len(terms) == 1:
            return terms[0]
        else:
            return And(terms.pop(), MaplCirup.big_and(terms))

    def value_iteration(self, discount: float = None, error: float = None, horizon: int = None) -> None:
        starttime_vi = time.time()

        if discount is not None:
            self._discount = discount
            self._ddc.set_discount(self._discount)

        if error is not None:
            self._error = error

        if horizon is not None:
            self._horizon = horizon

        utility = tf.zeros(2**len(self._state_vars), dtype=tf.float32)
        while True:
            if self._discount == 1 or horizon is not None:  # loop for horizon length
                if self._iterations_count >= self._horizon:
                    break

            new_utility = self._ddc.max_eu(utility)

            delta = tf.norm(new_utility-utility, ord=np.inf)
            utility = new_utility
            self._iterations_count += 1

            print('Iteration ' + str(self._iterations_count) + ' with delta: ' + str(delta))

            if self._discount < 1:
                if horizon is not None:
                    # if the horizon is set, loop for horizon length (with discount)
                    if self._iterations_count >= self._horizon:
                        break
                else:
                    # loop until convergence
                    if delta <= self._error:  # * (1-self._discount) / self._discount:
                        break

        endtime_vi = time.time()
        self._vi_time = endtime_vi - starttime_vi

        u_idx = 0
        for u in utility:
            self._ddc.set_utility_label('u' + str(u_idx), self._discount * u)
            u_idx += 1

    def print_explicit_policy(self) -> None:
        print("\nPOLICY FUNCTION:\n")
        state = self._state_vars
        while state:
            # collect state evidence
            state_evidence: Dict[str, bool] = dict()
            for term in state:
                term_var = str(term.args[0] if isinstance(term, Not) else term)
                state_evidence[term_var] = False if isinstance(term, Not) else True

            _, eu, decisions = self._ddc.best_dec(state_evidence)

            print(str(state) + ' -> ' + str(decisions) + ' (eu: ' + str(eu) + ')')
            state = MaplCirup.enumeration_next(state)

    def set_horizon(self, horizon: int) -> None:
        self._horizon = horizon

    def set_discount_factor(self, discount: float) -> None:
        self._discount = discount

    def size(self) -> int:
        """
        Returns the size of the circuit.
        """
        return self._ddc.size()

    def iterations(self) -> int:
        return self._iterations_count

    def compile_time(self) -> float:
        """
        Returns the amount of time required for compilation.
        """
        return self._compile_time

    def minimize_time(self) -> float:
        """
        Returns the amount of time required for compilation.
        """
        return self._minimize_time

    def value_iteration_time(self) -> float:
        return self._vi_time

    def tot_time(self) -> float:
        return self._compile_time + self._minimize_time + (self._vi_time if self._vi_time is not None else 0)

    def view_dot(self) -> None:
        """
        View the dot representation of the transition circuit.
        """
        self._ddc.view_dot()
