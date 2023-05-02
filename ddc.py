"""
Dynamic Decision Circuit (DDC)
"""
import numpy as np
from typing import List, Dict, Set
from enum import Enum
from collections import namedtuple
import graphviz
import typing

import numba

from semiring import BestDecSemiring, Label, MEUSemiring

from problog.logic import Term, Constant, Not
from problog.sdd_formula_explicit import SDDExplicit
from pysdd.sdd import SddNode

VarIndex = namedtuple("VarIndex", "pos, neg")


class DDC:
    """
    Class for dynamic decision circuits (DDCs).
    """

    _false = -1
    _true = -1
    _decisions: Set[int] = set()
    _semiring = None
    _id = 1
    _reuse_and_nodes_counter = 0

    def __init__(self):
        self._root: int = self._false
        self._children: Dict[int, List[int]] = dict()
        self._type: Dict[int, NodeType] = dict()
        self._var2node: Dict[str, VarIndex] = dict()
        self._label: Dict[int, Label] = dict()
        self._cache: Dict[int, Label] = dict()
        self._state_vars: List[str] = []
        self._states: Dict[int, np.array] = dict()
        self._compact_and_nodes = False
        self._reuse_and_nodes = True
        self._ands: Dict[(int, int), int] = dict()

    @classmethod
    def create_from(
        cls, sdd: SDDExplicit, state_vars: List[Term], rewards: Dict[Term, Constant]
    ):
        root: SddNode = sdd.get_root_inode()

        ddc = cls()
        ddc._state_vars = [str(x) for x in state_vars]

        # Retrieve variable names
        literal_id2name: Dict[int, List[str]] = dict()
        var_count = sdd.get_manager().varcount
        for name, key in sdd.get_names():
            assert (
                key >= 0 or abs(key) in sdd.atom2var
            ), "Named variable with negative key not existing as positive"
            inode_index = sdd.atom2var.get(key, -1)
            if 0 <= inode_index <= var_count:
                if inode_index in literal_id2name:
                    literal_id2name[inode_index].append(str(name))
                else:
                    literal_id2name[inode_index] = [str(name)]

        # Create DDC from SDD
        ddc._root = ddc._compact_sdd(root, literal_id2name, dict())

        # Init labelling function
        if ddc._true != ddc._false:
            # True node has been initialised, this, set the label
            ddc._label[ddc._true] = Label(1.0, 0.0, set())

        # set neutral prior weights
        for var in ddc._state_vars:
            ddc._set_positive_label(var, Label(1.0, 0.0, set()))
            ddc._set_negative_label(var, Label(1.0, 0.0, set()))

        # set all the other weights from the SDD
        weights: Dict[int, Term] = dict(sdd.get_weights())
        for key in weights:
            index = sdd.atom2var.get(abs(key), -1)
            assert index != -1, "Weighted variable missing"
            if isinstance(weights[key], bool) and weights[key] is True:
                if key > 0:
                    ddc._set_positive_label(
                        literal_id2name[index][0], Label(1.0, 0.0, set())
                    )
                else:
                    ddc._set_negative_label(
                        literal_id2name[index][0], Label(1.0, 0.0, set())
                    )
            elif isinstance(
                weights[key], Constant
            ):  # probability (utilities at the beginning are all zero)
                prob: float = weights[key].compute_value()
                ddc._set_pn_labels(
                    literal_id2name[index][0], key > 0, Label(prob, 0.0, set())
                )
            elif weights[key] == Term("?"):  # decision
                node_id = ddc._var2node[literal_id2name[index][0]]
                ddc._decisions.add(node_id.pos)
                ddc._decisions.add(node_id.neg)
                ddc._set_positive_label(
                    literal_id2name[index][0], Label(1.0, 0.0, {node_id.pos})
                )
                ddc._set_negative_label(
                    literal_id2name[index][0], Label(1.0, 0.0, {node_id.neg})
                )

        # for all the other literals
        for node_id, node_type in ddc._type.items():
            if node_type == NodeType.LITERAL and node_id not in ddc._label:
                ddc._label[node_id] = Label(1.0, 0.0, set())

        # init reward parameters
        for r in rewards:
            if isinstance(r, Not):
                key = -sdd.get_node_by_name(r.args[0])
                r = r.args[0]
            else:
                key = sdd.get_node_by_name(r)

            if key is None:  # if it is None it means r is not in the circuit
                continue

            ddc._init_reward_label(str(r), key > 0, rewards[r].compute_value())

        # Add named variables with negative key that where not considered above
        # note that each variable with a negative key at this point is already added in its positive key version
        for name, key in sdd.queries():
            if key < 0:
                var_name = str(name)
                assert (
                    var_name not in ddc._var2node
                ), "Variable with negative key already in the DDC"
                inode_index = sdd.atom2var.get(abs(key), -1)
                assert inode_index != -1, "Negative query variable missing"
                ddc_index = ddc._var2node[literal_id2name[inode_index][0]]
                assert (
                    ddc_index.pos != ddc._false
                ), "Query val with positive index set to false"
                ddc._var2node[var_name] = VarIndex(ddc_index.neg, ddc_index.pos)

        # Create vectorised evidence for state variables
        var_num: int = len(ddc._state_vars)
        rep: int = 0
        for var in ddc._state_vars:
            var_num -= 1
            index = ddc._var2node[var]
            ddc._states[index.pos] = np.tile(
                np.repeat(np.array([1, 0]), 2**rep), 2**var_num
            )
            ddc._states[index.neg] = np.tile(
                np.repeat(np.array([0, 1]), 2**rep), 2**var_num
            )
            rep += 1

        return ddc

    def _compact_sdd(
        self, node: SddNode, lit_name_map: Dict[int, List[str]], visited: Dict[int, int]
    ) -> int:
        if node.id in visited:
            assert (
                visited[node.id] in self._children
            ), "Compacting SDD removed wrong nodes"
            return visited[node.id]
        if node.is_literal():
            # A probabilistic rule '0.5::a :- b.' is split into:
            # a :- b, choice(x,g,a).
            # 0.5::choice(x,g,a).
            # where 'x' is a (line?) number, and 'g' is a group number.
            # TODO Or something similar. Understand why there are also 'body' literals.
            try:
                var_names = lit_name_map[abs(node.literal)]
            except KeyError:
                print("Literal id %s doesn't have a name." % node.literal)

            # If the leaf already exists
            var_name = var_names[0]  # I can just check the first one
            if var_name in self._var2node:
                if node.literal > 0:
                    if self._var2node[var_name].pos != self._false:
                        return self._var2node[var_name].pos
                else:
                    if self._var2node[var_name].neg != self._false:
                        return self._var2node[var_name].neg

            # Otherwise, insert it in the circuit
            node_id = self._id
            self._id += 1
            self._children[node_id] = []
            self._type[node_id] = NodeType.LITERAL
            visited[node.id] = node_id

            # update the var2node mapping
            for var_name in var_names:
                if var_name in self._var2node:
                    var_index = self._var2node[var_name]
                    if node.literal > 0:  # insert the index of the positive variable
                        self._var2node[var_name] = VarIndex(node_id, var_index.neg)
                    else:  # insert the index of the negated variable
                        self._var2node[var_name] = VarIndex(var_index.pos, node_id)
                else:
                    if node.literal > 0:
                        self._var2node[var_name] = VarIndex(node_id, self._false)
                    else:  # insert the index of the negated variable
                        self._var2node[var_name] = VarIndex(self._false, node_id)

            return node_id

        elif node.is_true():
            if self._true == self._false:
                # Initialise true node for the first time
                node_id = self._id
                self._id += 1
                self._children[node_id] = []
                self._type[node_id] = NodeType.TRUE
                visited[node.id] = node_id
                self._true = node_id
            return self._true
        elif node.is_false():
            return self._false
        elif node.is_decision():
            or_children = []
            for prime, sub in node.elements():
                # TODO AND nodes are apparently not cached..?
                sub_node = self._compact_sdd(sub, lit_name_map, visited)
                if sub_node != self._false:
                    prime_node = self._compact_sdd(prime, lit_name_map, visited)
                    assert (
                        prime_node != self._false
                    ), "Vincent was wrong: prime can be false."
                    # Create AND node
                    # If an AND node has an AND child, I can compact it
                    sub_children = [sub_node]
                    if self._compact_and_nodes and self._type[sub_node] == NodeType.AND:
                        sub_children = self._children[sub_node]
                        self._children.pop(sub_node)
                        self._type.pop(sub_node)
                    prime_children = [prime_node]
                    if (
                        self._compact_and_nodes
                        and self._type[prime_node] == NodeType.AND
                    ):
                        prime_children = self._children[prime_node]
                        self._children.pop(prime_node)
                        self._type.pop(prime_node)
                    node_id = None
                    if not self._compact_and_nodes and self._reuse_and_nodes:
                        # It makes more sense to do this only if we don't compact and nodes otherwise we have to be
                        # careful to not compact AND nodes that are shared (i.e. with more than one parent)
                        # TODO : Check if make sense to compact consecutive AND nodes where the child doesn't have any other parents
                        try:
                            node_id = self._ands[(prime_node, sub_node)]
                            self._reuse_and_nodes_counter += 1
                        except KeyError:
                            self._ands[(prime_node, sub_node)] = self._id
                        # for nid, children in self._children.items():
                        #     if children == sub_children + prime_children:
                        #         node_id = nid
                        #         self._reuse_and_nodes_counter += 1
                        #         break
                    if node_id is None:
                        node_id = self._id
                        self._id += 1
                        self._children[node_id] = sub_children + prime_children
                        self._type[node_id] = NodeType.AND
                    or_children.append(node_id)
            # Create OR node
            if len(or_children) == 0:
                return self._false
            elif len(or_children) == 1:
                return or_children[0]
            else:
                node_id = self._id
                self._id += 1
                self._children[node_id] = or_children
                self._type[node_id] = NodeType.OR
                visited[node.id] = node_id
                return node_id

        else:
            raise TypeError("Unknown type for node %s" % node)

    def view_dot(self) -> None:
        """
        View the dot representation of the transition circuit.
        """
        dot = self.to_dot()

        b = graphviz.Source(dot)
        b.view()

    def to_dot(self) -> str:
        dot = ["digraph sdd {", "overlap=false;"]
        for node, children in self._children.items():
            dot_node = ""
            if self._type[node] == NodeType.TRUE:
                dot_node = f'{node} [shape=rectangle,label="True"];'
            elif self._type[node] == NodeType.LITERAL:
                var_name = ", ".join(self._node_to_var(node))
                dec = self._label[node].dec
                dec_label = "{}"
                if len(dec) > 0:
                    decs = [x for d in dec for x in self._node_to_var(d)]
                    dec_label = "{" + ", ".join(decs) + "}"
                label = f"({round(self._label[node].prob, 2)}, {self._label[node].eu}, {dec_label})"
                dot_node = f'{node} [shape=rectangle,label="{var_name} : {label}"];'
            elif self._type[node] == NodeType.AND or self._type[node] == NodeType.OR:
                var_name = "+" if self._type[node] == NodeType.OR else "×"
                dot_node = f'{node} [label="{var_name}",shape=circle,style=filled,fillcolor=gray95];\n'
                dot_children = []
                for child in children:
                    dot_children.append(f"{node} -> {child} [arrowhead=none];")
                dot_node += "\n".join(dot_children)
            dot.append(dot_node)

        dot += ["}"]
        return "\n".join(dot)

    def _node_to_var(self, node_id: int) -> List[str]:
        literal = []
        for var, index in self._var2node.items():
            if node_id == index.pos and index.pos != self._false:
                literal.append(var)
            if node_id == index.neg and index.neg != self._false:
                literal.append("¬" + var)
        return literal

    def _set_positive_label(self, var: str, label: Label):
        index = self._var2node[var]
        if index.pos != self._false:
            self._label[index.pos] = label

    def _set_negative_label(self, var: str, label: Label):
        index = self._var2node[var]
        if index.neg != self._false:
            self._label[index.neg] = label

    def _set_pn_labels(self, var: str, positive: bool, label: Label):
        if positive:
            self._set_positive_label(var, label)
            if self._var2node[var].neg not in self._label:
                # If the complement is not in the labelling function already, insert it
                self._set_negative_label(
                    var, Label(1 - label.prob, label.eu, label.dec)
                )
        else:
            self._set_negative_label(var, label)
            if self._var2node[var].pos not in self._label:
                # If the complement is not in the labelling function already, insert it
                self._set_positive_label(
                    var, Label(1 - label.prob, label.eu, label.dec)
                )

    def _init_reward_label(self, var: str, positive: bool, val: float):
        index = self._var2node[var].pos if positive else self._var2node[var].neg
        assert index in self._label, "Reward label wrongly initialised"
        old_label = self._label[index]
        self._label[index] = Label(old_label.prob, old_label.prob * val, old_label.dec)

    def set_utility_label(self, var: str, eu: float):
        index = self._var2node[var].pos
        if index != self._false:
            assert index in self._label, "Utility label not existing"
            old_label = self._label[index]
            self._label[index] = Label(old_label.prob, eu, old_label.dec)

    def get_utility_label(self, var: str) -> float:
        index = self._var2node[var].pos
        if index != self._false:
            return self._label[self._var2node[var].pos].eu
        else:
            return 0.0

    def size(self) -> int:
        return len(self._children)

    def _evidence_to_label(self, var: str, value: bool) -> typing.Tuple[int, Label]:
        # if variable 'a' is set to True, I will set ¬a probability to 0, and vice versa
        node_index = self._var2node[var]
        node_id = node_index.neg if value else node_index.pos
        label = self._label[node_id]
        return node_id, Label(0.0, 0.0, label.dec)  # eu = p * util

    def max_eu(self, numba_structures, cache) -> np.array:
        children = numba_structures["children"]
        node_type = numba_structures["node_type"]
        one_p = numba_structures["one_p"]
        one_eu = numba_structures["one_eu"]
        one_max = numba_structures["one_max"]
        states = numba_structures["states"]
        label_p = numba_structures["label_p"]
        label_eu = numba_structures["label_eu"]
        label_max = numba_structures["label_max"]

        cache_p = cache["cache_p"]
        cache_eu = cache["cache_eu"]
        cache_max = cache["cache_max"]

        return _max_eu(
            self._root,
            children,
            node_type,
            one_p,
            one_eu,
            one_max,
            states,
            label_p,
            label_eu,
            label_max,
            cache_p,
            cache_eu,
            cache_max,
        )

    def update_utility_label(
        self, uindex_to_unode, new_utility, label_eu, discount, ones
    ):
        return _update_utility_label(
            uindex_to_unode, new_utility, label_eu, discount, ones
        )

    def best_dec(self, state: Dict[str, bool] = None) -> Label:
        return self._evaluate_root_iter(BestDecSemiring(self._decisions), state)

    def _evaluate_root_iter(
        self, semiring: BestDecSemiring, evidence: Dict[str, bool] = None
    ) -> Label:
        self._semiring = semiring

        evidence_label: Dict[int, Label] = dict()
        if evidence is not None:
            for var, value in evidence.items():
                node, label = self._evidence_to_label(var, value)
                evidence_label[node] = label

        self._cache = dict()
        for node, children in self._children.items():
            if self._type[node] == NodeType.TRUE:
                self._cache[node] = self._semiring.one()
            elif self._type[node] == NodeType.LITERAL:
                if evidence_label is not None and node in evidence_label:
                    self._cache[node] = evidence_label[node]
                else:
                    self._cache[node] = self._label[node]
            elif self._type[node] == NodeType.OR:
                assert (
                    len(self._children[node]) > 0
                ), "There is an OR node with no children"
                total = self._cache[children[0]]
                for child in children[1:]:
                    total = self._semiring.plus(total, self._cache[child])
                self._cache[node] = total
            elif self._type[node] == NodeType.AND:
                assert (
                    len(self._children[node]) > 0
                ), "There is an AND node with no children"
                total = self._cache[children[0]]
                for child in children[1:]:
                    total = self._semiring.times(total, self._cache[child])
                self._cache[node] = total

        ddc_eval = self._cache[self._root]
        (prob, eu, dec) = semiring.normalise(ddc_eval, ddc_eval)

        # turn decision ids into variable names (i.e. human-readable)
        dec_vars: Set[str] = set()
        for d in dec:
            for d_var in self._node_to_var(d):
                dec_vars.add(d_var)

        return Label(prob, eu, dec_vars)

    # def evaluate_root(self, semiring: BestDecSemiring, evidence: Dict[str, bool] = None) -> Label:
    #     self._semiring = semiring
    #
    #     evidence_label: Dict[int, Label] = dict()
    #     if evidence is not None:
    #         for var, value in evidence.items():
    #             node, label = self._evidence_to_label(var, value)
    #             evidence_label[node] = label
    #
    #     ddc_eval = self._evaluate_node(self._root, evidence_label)
    #     self._cache = dict()  # empty cache
    #     (prob, eu, dec) = semiring.normalise(ddc_eval, ddc_eval)
    #
    #     # turn decision ids into variable names (i.e. human-readable)
    #     dec_vars: Set[str] = set()
    #     for d in dec:
    #         for d_var in self._node_to_var(d):
    #             dec_vars.add(d_var)
    #
    #     return Label(prob, eu, dec_vars)
    #
    # def _evaluate_node(self, node: int, evidence_label: Dict[int, Label]) -> Label:
    #     assert node != self._false, "False node is evaluated"
    #     if node in self._cache:
    #         return self._cache[node]
    #     if self._type[node] == NodeType.TRUE:
    #         self._cache[node] = self._semiring.one()
    #         return self._semiring.one()
    #     elif self._type[node] == NodeType.LITERAL:
    #         res = self._label[node]
    #         if evidence_label is not None and node in evidence_label:
    #             res = evidence_label[node]
    #         self._cache[node] = res
    #         return res
    #     elif self._type[node] == NodeType.OR or self._type[node] == NodeType.AND:
    #         assert len(self._children[node]) > 0, "There is an AND/OR node with no children"
    #         total = self._evaluate_node(self._children[node][0], evidence_label)
    #         for child in self._children[node][1:]:
    #             child_eval = self._evaluate_node(child, evidence_label)
    #             new_total = self._semiring.plus(total, child_eval) if self._type[node] == NodeType.OR \
    #                 else self._semiring.times(total, child_eval)
    #             total = new_total
    #         self._cache[node] = total
    #         return total

    def impossible_utilities(self) -> List[str]:
        impossible_utilities = []
        for var, index in self._var2node.items():
            if index.pos == self._false:
                impossible_utilities.append(var)
        return impossible_utilities

    def print_info(self):
        print("Number of re-used AND nodes: %s" % self._reuse_and_nodes_counter)
        print(
            "Number of literals: %s"
            % len([x for x in self._children if self._type[x] == NodeType.LITERAL])
        )
        print(
            "Number of leaves: %s"
            % len([x for x in self._children if len(self._children[x]) == 0])
        )
        print(
            "Number of AND nodes: %s"
            % len([x for x in self._children if self._type[x] == NodeType.AND])
        )
        print(
            "Number of multiplications required: %s"
            % sum(
                [
                    len(self._children[x])
                    for x in self._children
                    if self._type[x] == NodeType.AND
                ]
            )
        )
        print(
            "Number of OR nodes: %s"
            % len([x for x in self._children if self._type[x] == NodeType.OR])
        )
        print(
            "Number of sum/max required: %s"
            % sum(
                [
                    len(self._children[x])
                    for x in self._children
                    if self._type[x] == NodeType.OR
                ]
            )
        )

        or_nodes_degree = [
            len(self._children[x])
            for x in self._children
            if self._type[x] == NodeType.OR
        ]
        and_nodes_degree = [
            len(self._children[x])
            for x in self._children
            if self._type[x] == NodeType.AND
        ]
        print("Max OR nodes degree: %s" % max(or_nodes_degree))
        mean = sum(or_nodes_degree) / len(or_nodes_degree)
        print("Average OR nodes degree: %s (+-%s)" % (mean, np.std(or_nodes_degree)))
        print("Max AND nodes degree: %s" % max(and_nodes_degree))
        mean = sum(and_nodes_degree) / len(and_nodes_degree)
        print("Average AND nodes degree: %s (+-%s)" % (mean, np.std(and_nodes_degree)))


class NodeType(Enum):
    TRUE = 1
    LITERAL = 2
    AND = 3
    OR = 4


@numba.njit(fastmath=True, parallel=False)
def _max_eu(
    root,
    children,
    node_type,
    one_p,
    one_eu,
    one_max,
    states,
    label_p,
    label_eu,
    label_max,
    cache_p,
    cache_eu,
    cache_max,
):
    for node in children.keys():
        if node_type[node] == 1:
            cache_p[node] = one_p
            cache_eu[node] = one_eu
            cache_max[node] = one_max
        elif node_type[node] == 2:
            p = label_p[node]
            eu = label_eu[node]
            m = label_max[node]
            if node in states:
                cache_p[node] = states[node]
                cache_eu[node] = eu * states[node]
                cache_max[node] = m
            else:
                cache_p[node] = p
                cache_eu[node] = eu
                cache_max[node] = m
        else:
            node_children = children[node]
            total_p = cache_p[node_children[0]]
            total_eu = cache_eu[node_children[0]]
            total_max = cache_max[node_children[0]]
            if node_type[node] == 3:
                for i in range(1, len(node_children)):
                    node_c = node_children[i]
                    c_p = cache_p[node_c]
                    c_eu = cache_eu[node_c]
                    c_max = cache_max[node_c]

                    total_eu = total_p * c_eu + c_p * total_eu
                    total_p = total_p * c_p
                    total_max = total_max or c_max
            elif node_type[node] == 4:
                for i in range(1, len(node_children)):
                    node_c = node_children[i]
                    c_p = cache_p[node_c]
                    c_eu = cache_eu[node_c]
                    c_max = cache_max[node_c]

                    if total_max or c_max:
                        take_total = (c_p == 0) | (
                            (total_p != 0) & (total_eu / total_p > c_eu / c_p)
                        )
                        total_p = (total_p * take_total) + (
                            c_p * np.logical_not(take_total)
                        )
                        total_eu = (total_eu * take_total) + (
                            c_eu * np.logical_not(take_total)
                        )
                        total_max = True
                    else:
                        total_p = total_p + c_p
                        total_eu = total_eu + c_eu
                        total_max = False

            cache_p[node] = total_p
            cache_eu[node] = total_eu
            cache_max[node] = total_max

    root_p = cache_p[root]
    root_eu = cache_eu[root]

    eu = root_eu / root_p

    return eu


@numba.njit(fastmath=True, parallel=False)
def _update_utility_label(uindex_to_unode, new_utility, label_eu, discount, ones):
    for i in range(new_utility.shape[0]):
        unode = uindex_to_unode[i]
        label_eu[unode] = (discount * new_utility[i]) * ones

    return label_eu
