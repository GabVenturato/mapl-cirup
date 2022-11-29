"""
Dynamic Decision Circuit (DDC)
"""

from typing import List, Dict, Set
from enum import Enum
from collections import namedtuple
import graphviz

from problog.logic import Term, Constant, Not
from problog.sdd_formula_explicit import SDDExplicit
from pysdd.sdd import SddNode

Label = namedtuple('Label', 'prob, util, dec')
VarIndex = namedtuple('VarIndex', 'pos, neg')


class MaxEUSemiring:
    _empty_set = set()

    def __init__(self, decisions):
        self.val_zero = Label(0.0, 0.0, decisions)
        self.zero_decision_length = len(decisions)

    def one(self):
        return Label(1.0, 0.0, self._empty_set)

    def zero(self):
        return self.val_zero

    def plus(self, a: Label, b: Label):
        p_a, eu_a, d_a = a
        p_b, eu_b, d_b = b
        if len(d_a) or len(d_b):
            # Max
            if len(d_b) == self.zero_decision_length:  # avoid false
                result = a
            elif len(d_a) == self.zero_decision_length:  # avoid false
                result = b
            elif p_a == 0:
                result = b
            elif p_b == 0:
                result = a
            elif eu_a / p_a >= eu_b / p_b:  # check on eu
                result = a
            else:
                result = b
            print("max( (%s, %s, %s), (%s, %s, %s) ) = (%s, %s, %s)" %
                  (a.prob, a.util, a.dec, b.prob, b.util, b.dec, result.prob, result.util, result.dec))
            return result
        else:
            # Sum
            print("(%s, %s, %s) + (%s, %s, %s) ) = (%s, %s, %s)" %
                  (a.prob, a.util, a.dec, b.prob, b.util, b.dec, p_a + p_b, eu_a + eu_b, self._empty_set))
            return Label(p_a + p_b, eu_a + eu_b, self._empty_set)

    def times(self, a: Label, b: Label):
        p_a, eu_a, d_a = a
        p_b, eu_b, d_b = b
        eu_n = p_a * eu_b + p_b * eu_a
        print("(%s, %s, %s) * (%s, %s, %s) ) = (%s, %s, %s)" %
              (a.prob, a.util, a.dec, b.prob, b.util, b.dec, p_a * p_b, eu_n, d_a.union(d_b)))
        return Label(p_a * p_b, eu_n, d_a.union(d_b))


class DDC:
    """
    Class for dynamic decision circuits (DDCs).
    """
    _false = -1
    _decisions: Set[int] = set()
    _semiring = None
    _id = 1

    def __init__(self):
        self._root: int = self._false
        self._children: Dict[int, List[int]] = {0: []}
        self._type: Dict[int, NodeType] = {0: NodeType.TRUE}
        self._var2node: Dict[str, VarIndex] = {'true': VarIndex(0, 0)}
        self._label: Dict[int, Label] = dict()

        self._label[-1] = Label(0, 0, set())  # set label for False
        self._label[0] = Label(1, 0, set())  # set label for True

    @classmethod
    def create_from(cls, sdd: SDDExplicit, rewards: Dict[Term, Constant]):
        root: SddNode = sdd.get_root_inode()

        ddc = cls()

        # Retrieve variable names
        literal_id2name: Dict[int, List[str]] = dict()
        var_count = sdd.get_manager().varcount
        for (name, index) in sdd.get_names():
            inode_index = sdd.atom2var.get(index, -1)
            if 0 <= inode_index <= var_count:
                if inode_index in literal_id2name:
                    literal_id2name[inode_index].append(str(name))
                else:
                    literal_id2name[inode_index] = [str(name)]

        # Create DDC from SDD
        ddc._root = ddc._compact_sdd(root, literal_id2name, dict())

        # Init labelling function
        weights: Dict[int, Term] = dict(sdd.get_weights())

        for key in weights:
            index = sdd.atom2var.get(abs(key), -1)
            if isinstance(weights[key], bool) and weights[key] is True:
                if key > 0:
                    ddc._set_positive_label(literal_id2name[index][0], Label(1.0, 0.0, set()))
                else:
                    ddc._set_negative_label(literal_id2name[index][0], Label(1.0, 0.0, set()))
            elif isinstance(weights[key], Constant):  # probability (utilities at the beginning are all zero)
                prob: float = weights[key].compute_value()
                ddc._set_pn_labels(literal_id2name[index][0], key > 0, Label(prob, 0.0, set()))
            elif weights[key] == Term("?"):  # decision
                node_id = ddc._var2node[literal_id2name[index][0]]
                ddc._decisions.add(node_id.pos)
                ddc._decisions.add(node_id.neg)
                ddc._set_positive_label(literal_id2name[index][0], Label(1.0, 0.0, {node_id.pos}))
                ddc._set_negative_label(literal_id2name[index][0], Label(1.0, 0.0, {node_id.neg}))

        # for reward parameters
        for r in rewards:
            if isinstance(r, Not):
                key = -sdd.get_node_by_name(r.args[0])
                r = r.args[0]
            else:
                key = sdd.get_node_by_name(r)

            if key is None:  # if it is None it means r is not in the circuit
                continue

            if key > 0:
                ddc._update_util_positive_label(str(r), rewards[r].compute_value())
            else:
                ddc._update_util_negative_label(str(r), rewards[r].compute_value())

        # for all the other literals
        for node_id, node_type in ddc._type.items():
            if node_type == NodeType.LITERAL and node_id not in ddc._label:
                ddc._label[node_id] = Label(1.0, 0.0, set())

        return ddc

    def _compact_sdd(self, node: SddNode, lit_name_map: Dict[int, List[str]], visited: Dict[int, int]) -> int:
        if node.id in visited:
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
            return 0
        elif node.is_false():
            return self._false
        elif node.is_decision():
            or_children = []
            for (prime, sub) in node.elements():
                # TODO AND nodes are apparently not cached..?
                sub_node = self._compact_sdd(sub, lit_name_map, visited)
                if sub_node != self._false:
                    prime_node = self._compact_sdd(prime, lit_name_map, visited)
                    assert prime_node != self._false, "Vincent was wrong: prime can be false."
                    # Create AND node
                    # TODO Check if it's ok to remove AND nodes (tricky if an AND node is child of both an AND and OR)
                    sub_children = [sub_node]
                    if self._type[sub_node] == NodeType.AND:
                        sub_children = self._children[sub_node]
                        self._children.pop(sub_node)
                        self._type.pop(sub_node)
                    prime_children = [prime_node]
                    if self._type[prime_node] == NodeType.AND:
                        prime_children = self._children[prime_node]
                        self._children.pop(prime_node)
                        self._type.pop(prime_node)
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
            raise TypeError('Unknown type for node %s' % node)

    def view_dot(self) -> None:
        """
        View the dot representation of the transition circuit.
        """
        dot = self.to_dot()

        b = graphviz.Source(dot)
        b.view()

    def to_dot(self) -> str:
        dot = [
            "digraph sdd {",
            "overlap=false;"
        ]
        for node, children in self._children.items():
            dot_node = ""
            if self._type[node] == NodeType.TRUE:
                dot_node = f"{node} [shape=rectangle,label=\"True\"];"
            elif self._type[node] == NodeType.LITERAL:
                var_name = ", ".join(self._node_to_var(node))
                dec = self._label[node].dec
                dec_label = "{}"
                if len(dec) > 0:
                    decs = self._node_to_var(dec.pop())
                    dec_label = "{" + ", ".join(decs) + "}"
                label = f"({round(self._label[node].prob, 2)}, {self._label[node].util}, {dec_label})"
                dot_node = f"{node} [shape=rectangle,label=\"{var_name} : {label}\"];"
            elif self._type[node] == NodeType.AND or self._type[node] == NodeType.OR:
                var_name = '+' if self._type[node] == NodeType.OR else '×'
                dot_node = f"{node} [label=\"{var_name}\",shape=circle,style=filled,fillcolor=gray95];\n"
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
            if node_id == index.pos:
                literal.append(var)
            if node_id == index.neg:
                literal.append("¬" + var)
        return literal

    def _set_positive_label(self, var: str, label: Label):
        index = self._var2node[var]
        self._label[index.pos] = label

    def _set_negative_label(self, var: str, label: Label):
        index = self._var2node[var]
        self._label[index.neg] = label

    def _set_pn_labels(self, var: str, positive: bool, label: Label):
        if positive:
            self._set_positive_label(var, label)
            self._set_negative_label(var, Label(1-label.prob, label.util, label.dec))
        else:
            self._set_negative_label(var, label)
            self._set_positive_label(var, Label(1-label.prob, label.util, label.dec))

    def _update_util_positive_label(self, var: str, util: float):
        index = self._var2node[var]
        old_label = self._label[index.pos]
        self._label[index.pos] = Label(old_label.prob, util, old_label.dec)

    def _update_util_negative_label(self, var: str, util: float):
        index = self._var2node[var]
        old_label = self._label[index.neg]
        self._label[index.neg] = Label(old_label.prob, util, old_label.dec)

    def size(self) -> int:
        return len(self._children)

    def _evidence_to_label(self, var: str, value: bool) -> (int, Label):
        # if variable 'a' is set to True, I will set ¬a probability to 0, and vice versa
        node_index = self._var2node[var]
        node_id = node_index.pos if value else node_index.neg
        label = self._label[node_id]
        return node_id, Label(0.0, label.util, label.dec)

    def maxeu(self, state: Dict[str, bool]) -> Label:
        return self.evaluate(MaxEUSemiring(self._decisions), state)

    def evaluate(self, semiring: MaxEUSemiring, evidence: Dict[str, bool] = None) -> Label:
        self._semiring = semiring

        evidence_label: Dict[int, Label] = dict()
        if evidence is not None:
            for var, value in evidence.items():
                node, label = self._evidence_to_label(var, value)
                evidence_label[node] = label

        return self._evaluate_node(self._root, evidence_label)

    def _evaluate_node(self, node: int, evidence_label: Dict[int, Label]) -> Label:
        if self._type[node] == NodeType.TRUE:
            return self._semiring.one()
        elif self._type[node] == NodeType.LITERAL:
            if evidence_label is not None and node in evidence_label:
                return evidence_label[node]
            return self._label[node]
        elif self._type[node] == NodeType.OR:
            total = self._semiring.zero()
            for child in self._children[node]:
                child_eval = self._evaluate_node(child, evidence_label)
                new_total = self._semiring.plus(total, child_eval)
                total = new_total
            return total
        elif self._type[node] == NodeType.AND:
            total = self._semiring.one()
            for child in self._children[node]:
                child_eval = self._evaluate_node(child, evidence_label)
                new_total = self._semiring.times(total, child_eval)
                total = new_total
            return total


class NodeType(Enum):
    TRUE = 1
    LITERAL = 2
    AND = 3
    OR = 4

# class Label:
#     def __init__(self, prob: float, util: float, dec: Set[str]):  # TODO float must be changed to some Numpy format
#         self._prob = prob
#         self._util = util
#         self._dec = dec


