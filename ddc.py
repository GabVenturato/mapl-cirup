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


class DDC:
    """
    Class for dynamic decision circuits (DDCs).
    """
    _false = -1

    def __init__(self):
        self._root: int = self._false
        self._children: List[List[int]] = [[]]
        self._type: List[NodeType] = [NodeType.TRUE]
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
        ddc._root = ddc._compact_sdd(root, literal_id2name)

        # Init labelling function
        weights: Dict[int, Term] = dict(sdd.get_weights())

        for key in weights:
            index = sdd.atom2var.get(abs(key), -1)
            if isinstance(weights[key], bool) and weights[key] is True:
                ddc._set_pn_labels(literal_id2name[index][0], key > 0, Label(1, 0, set()))
            elif isinstance(weights[key], Constant):  # probability (utilities at the beginning are all zero)
                prob: float = weights[key].compute_value()
                ddc._set_pn_labels(literal_id2name[index][0], key > 0, Label(prob, 0, set()))
            elif weights[key] == Term("?"):  # decision
                node_id = ddc._var2node[literal_id2name[index][0]]
                ddc._set_positive_label(literal_id2name[index][0], Label(1, 0, {node_id.pos}))
                ddc._set_negative_label(literal_id2name[index][0], Label(1, 0, {node_id.neg}))

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
        for node_id, node_type in enumerate(ddc._type):
            if node_type == NodeType.LITERAL and node_id not in ddc._label:
                ddc._label[node_id] = Label(1, 0, set())

        return ddc

    def _compact_sdd(self, node: SddNode, lit_name_map: Dict[int, List[str]]) -> int:
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
            self._children.append([])
            self._type.append(NodeType.LITERAL)
            node_id: int = len(self._children) - 1

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
                sub_node = self._compact_sdd(sub, lit_name_map)
                if sub_node != self._false:
                    prime_node = self._compact_sdd(prime, lit_name_map)
                    # TODO check if prime node is false?
                    # Create AND node
                    self._children.append([sub_node, prime_node])
                    self._type.append(NodeType.AND)
                    or_children.append(len(self._children) - 1)
            # Create OR node
            if len(or_children) == 0:
                return self._false
            elif len(or_children) == 1:
                return or_children[0]
            else:
                self._children.append(or_children)
                self._type.append(NodeType.OR)
                return len(self._children) - 1

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
        for node, children in enumerate(self._children):
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
                label = f"({self._label[node].prob}, {self._label[node].util}, {dec_label})"
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
            self._set_positive_label(var, Label(1 - label.prob, label.util, label.dec))

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


