"""
MArkov PLanning with CIRcuit bellman UPdates (mapl-cirup)
"""
import copy
import os
import tempfile
import re
import time
import graphviz
import random

from typing import Dict, Set, List

from problog import get_evaluatable
from problog.clausedb import ClauseDB
from problog.formula import LogicFormula
from problog.logic import Term, Constant, Clause, Not, And
from problog.program import PrologFile, PrologString
from problog.engine import DefaultEngine
from problog.sdd_formula_explicit import SDDExplicit, x_constrained_named

from semiring import SemiringMAXEU, SemiringActiveUtilities, pn_weight
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
    _horizon = 1                                                # Default future lookahead
    _discount = 1                                               # Default discount factor
    _error = 0.01                                               # Default maximum error allowed for value iteration
    _utilities: Dict[Term, List[Term]] = {}                     # added (expected) utility parameters with their state
    _labels: Dict[int, pn_weight] = {}                          # label function for che circuit
    _iterations_count: int = 0                                  # number of iterations for policy convergence
    _vi_time = None                                             # time required for value iteration

    def __init__(self, filename):
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
        self._circuit = self._compilation(grounded_prog)
        endtime_compilation = time.time()
        self._compile_time = endtime_compilation - starttime_compilation
        print("Compilation done! (circuit size: %s)" % self.size())

        print("Minimizing...")
        starttime_minimization = time.time()
        # self._minimize()
        endtime_minimization = time.time()
        self._minimize_time = endtime_minimization - starttime_minimization
        print("Minimization done! (circuit size: %s)" % self.size())

        self._ddc: DDC = DDC.create_from(self._circuit, self._rewards)
        print("DDC size: %s" % self._ddc.size())
        # self._ddc.view_dot()

        self._semiring = self._get_semiring()

        return

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
            if new_var.probability is None:
                new_var.probability = Constant(0.5)
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

    def _minimize(self) -> None:
        """
        SDD Minimization (Sec 5.5 of the advanced manual).
        """
        # If one wants to limit times more strictly. Default parameters should be: 180, 60, 30, 10.
        # self._circuit.get_manager().get_manager().set_vtree_search_time_limit(60)
        # self._circuit.get_manager().get_manager().set_vtree_fragment_time_limit(20)
        # self._circuit.get_manager().get_manager().set_vtree_operation_time_limit(10)
        # self._circuit.get_manager().get_manager().set_vtree_apply_time_limit(5)

        # The following call to 'ref()' is required otherwise the minimization removes necessary nodes
        self._circuit.get_root_inode().ref()
        self._circuit.get_manager().get_manager().minimize_limited()

    def _get_semiring(self) -> (SemiringMAXEU, SemiringMAXEU):
        """
        Initialize the maximum expected utility semiring (MEU).
        :return: The MEU semiring.
        """
        decisions_keys = set()
        for decision in self._decisions:
            decisions_keys.add(self._circuit.get_node_by_name(decision))
        semiring = SemiringMAXEU(decisions_keys)

        return semiring

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
        self._labels = self._init_utility()

        if discount is not None:
            self._discount = discount

        if error is not None:
            self._error = error

        if horizon is not None:
            self._horizon = horizon

        while True:
            if self._discount == 1:  # loop for horizon length
                if self._iterations_count >= self._horizon:
                    break

            delta = self._update_utility(self._labels)
            self._iterations_count += 1

            print('Iteration ' + str(self._iterations_count) + ' with delta: ' + str(delta))

            if self._discount < 1:  # loop until convergence
                if delta <= self._error * (1-self._discount) / self._discount:
                    break

        endtime_vi = time.time()
        self._vi_time = endtime_vi - starttime_vi

    def _init_utility(self) -> Dict[int, pn_weight]:
        labels: Dict[int, pn_weight] = dict()
        weights: Dict[int, Term] = dict(self._circuit.get_weights())

        # set label for True node
        # use '%' as a placeholder name for terms in pn_weight because it can't come from the model
        # TODO: Is this necessary?
        labels[0] = pn_weight(Term('%', 1.0, Constant(0), set()), Term('%', 0.0, Constant(0), set()))

        # set labels for the others, except reward and utility parameters
        for key in weights:
            abs_key = abs(key)
            if isinstance(weights[key], Constant):  # probability
                prob: float = weights[key].compute_value()
                self._set_pn_weight(labels, key,
                                    Term('%', prob, Constant(0), set()),
                                    Term('%', 1 - prob, Constant(0), set()))

            elif weights[key] == self._decision_term:  # decision
                self._set_pn_weight(labels, key,
                                    Term('%', 1.0, Constant(0), {abs_key}),
                                    Term('%', 1.0, Constant(0), {-abs_key}))

        # set labels for reward parameters
        for r in self._rewards:
            if isinstance(r, Not):
                key = -self._circuit.get_node_by_name(r.args[0])
            else:
                key = self._circuit.get_node_by_name(r)

            if key is None:  # if it is None it means r is not in the circuit
                continue

            if abs(key) in labels:   # update the label
                MaplCirup._update_utility_label(labels, key, self._rewards[r].compute_value())
            else:   # insert the label
                self._set_pn_weight(labels, key,
                                    Term('%', 1.0, Constant(self._rewards[r].compute_value()), set()),
                                    Term('%', 1.0, Constant(0), set()))

        # set labels for utility parameters (all zero at the beginning)
        for u in self._utilities:
            key = self._circuit.get_node_by_name(u)

            if key is None:  # if it is None it means r is not in the circuit
                continue

            labels[abs(key)] = pn_weight(Term('%', 1.0, Constant(0), set()), Term('%', 1.0, Constant(0), set()))

        return labels

    @staticmethod
    def _set_pn_weight(labels: Dict[int, pn_weight], key: int, pos_w: Term, neg_w: Term) -> None:
        if key > 0:
            labels[abs(key)] = pn_weight(pos_w, neg_w)
        else:
            labels[abs(key)] = pn_weight(neg_w, pos_w)

    def _update_utility(self, labels: Dict[int, pn_weight]) -> float:
        # Compute the new utility values
        updated_utilities: Dict[Term, float] = dict()

        for u, state in self._utilities.items():
            # collect evidence
            evidence: Set[(Term, int, bool)] = set()
            for term in state:
                if isinstance(term, Not):
                    term_key = self._circuit.get_node_by_name(term.args[0])
                else:
                    term_key = self._circuit.get_node_by_name(term)

                clean_term = term.args[0] if isinstance(term, Not) else term
                bool_term = False if isinstance(term, Not) else True
                evidence.add((clean_term, term_key, bool_term))

            # set the evidence
            for term, key, val in evidence:
                self._circuit.add_evidence(term, key, val)

            # circuit evaluation
            _, eu, _ = self._circuit.evaluate(
                index=self._circuit.get_node_by_name(self._true_term),
                semiring=self._semiring,
                weights=labels
            )
            updated_utilities[u] = eu

            # clear evidence
            self._circuit.clear_evidence()

        # Update the label function
        utility_delta = 0
        for u in self._utilities:
            key = self._circuit.get_node_by_name(u)

            pos_weight, neg_weight = labels[abs(key)]
            old_utility_val = pos_weight.args[1].compute_value() if key > 0 else neg_weight.args[1].compute_value()

            MaplCirup._update_utility_label(labels, key, self._discount * updated_utilities[u])

            pos_weight, neg_weight = labels[abs(key)]
            new_utility_val = pos_weight.args[1].compute_value() if key > 0 else neg_weight.args[1].compute_value()

            if abs(new_utility_val - old_utility_val) > utility_delta:
                utility_delta = abs(new_utility_val - old_utility_val)

        return utility_delta

    @staticmethod
    def _update_utility_label(labels, key, val) -> None:
        pos_weight, neg_weight = labels[abs(key)]
        if key > 0:
            pos_weight = Term('%',
                              pos_weight.args[0],
                              Constant(val),
                              pos_weight.args[2]
                              )
        else:
            neg_weight = Term('%',
                              neg_weight.args[0],
                              Constant(val),
                              neg_weight.args[2]
                              )
        labels[abs(key)] = pn_weight(pos_weight, neg_weight)

    def print_explicit_policy(self) -> None:
        print("\nPOLICY FUNCTION:\n")
        state = self._state_vars
        while state:
            # collect evidence
            evidence: Set[(Term, int, bool)] = set()
            for term in state:
                if isinstance(term, Not):
                    term_key = self._circuit.get_node_by_name(term.args[0])
                else:
                    term_key = self._circuit.get_node_by_name(term)
                clean_term = term.args[0] if isinstance(term, Not) else term
                bool_term = False if isinstance(term, Not) else True
                evidence.add((clean_term, term_key, bool_term))

            # set the evidence
            for term, key, val in evidence:
                self._circuit.add_evidence(term, key, val)

            # circuit evaluation
            _, eu, best_decision_keys = self._circuit.evaluate(
                index=self._circuit.get_node_by_name(self._true_term),
                semiring=self._semiring,
                weights=self._labels
            )

            decisions = []
            for decision in self._decisions:
                decision_key = self._circuit.get_node_by_name(decision)
                for best_key in best_decision_keys:
                    if abs(best_key) == abs(decision_key):
                        if best_key >= 0:
                            decisions.append(decision)

            print(str(state) + ' -> ' + str(decisions) + ' (eu: ' + str(eu) + ')')
            state = MaplCirup.enumeration_next(state)

            # clear evidence
            self._circuit.clear_evidence()

    def set_horizon(self, horizon: int) -> None:
        self._horizon = horizon

    def set_discount_factor(self, discount: float) -> None:
        self._discount = discount

    # def set_selection_limit(self, limit) -> None:
    #     self._selection_limit = limit

    def size(self) -> int:
        """
        Compute the size of the circuit. Where the size is the size of the live nodes in the SDD circuit.
        For more info on "live nodes" see the advanced manual of the sdd package: http://reasoning.cs.ucla.edu/sdd/
        :return: The size of the circuit
        """
        return self._circuit.get_manager().get_manager().live_size()

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
        return self._compile_time + self._minimize_time + self._vi_time

    def view_dot(self) -> None:
        """
        View the dot representation of the transition circuit.
        """
        self._circuit.get_manager().get_manager().garbage_collect()  # .minimize()
        dot = self._circuit.sdd_to_dot(node=None, litnamemap=True)
        b = graphviz.Source(dot)
        b.view()
