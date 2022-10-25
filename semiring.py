"""
Definition of the semirings used by the tool.
"""

import math

from collections import namedtuple
from random import randint
from typing import Set

from problog.evaluator import Semiring, OperationNotSupported
from problog.logic import Constant, Term

pn_weight = namedtuple('pos_neg_weight', 'p_weight, n_weight')


class SemiringMAXEU(Semiring):
    """
    A pseudo semiring to maximise the expected utility. It is only a semiring within an X-constrained setting.
    Each element of this semiring is a triple (prob, eu, decision_set) where the last element is used to keep track
    of the decisions made so far in this node. Evidently, a times will result in union. A plus results in choosing
    between different decision sets and thus a max-like operation.
    """
    # element = (prob, eu, decision_set)

    def __init__(self, decisions):
        """
        :param decisions: A set of all possible positive decision keys
        :type decisions: set[int]
        """
        Semiring.__init__(self)
        all_decisions = {-x for x in decisions}
        all_decisions |= decisions
        self.val_zero = (0.0, 0.0, all_decisions)
        self.zero_decision_length = len(all_decisions)

    def one(self):
        return 1.0, 0.0, set()

    def zero(self):
        return self.val_zero

    def is_one(self, value):
        p, eu, d = value
        return 1.0 - 1e-12 < p < 1.0 + 1e-12 and 1.0 - 1e-12 < eu < 1.0 + 1e-12 and len(d) == 0

    def is_zero(self, value):
        p, eu, d = value
        return -1e-12 < p < 1e-12 and -1e-12 < eu < 1e-12

    def plus(self, a, b):
        p_a, eu_a, d_a = a
        p_b, eu_b, d_b = b
        if len(d_a) or len(d_b):    # 'or' because of ad_complement which has empty decision set.
            if len(d_b) == self.zero_decision_length:  # avoid false
                result = a
            elif len(d_a) == self.zero_decision_length:  # avoid false
                result = b
            elif p_a == 0:
                result = b
            elif p_b == 0:
                result = a
            elif eu_a / p_a >= eu_b / p_b:     # check on eu
                result = a
            else:
                result = b
            # print("MAX a %s b %s = %s" % (a,b,result))
            return result
        else:
            result = p_a + p_b, eu_a + eu_b, set()
            # print("plus a %s b %s = %s" % (a, b, result))
            return result

    def times(self, a, b):
        p_a, eu_a, d_a = a
        p_b, eu_b, d_b = b
        eu_n = p_a * eu_b + p_b * eu_a
        result = p_a * p_b, eu_n, d_a.union(d_b)
        # print("times a %s b %s = %s" % (a, b, result))
        return result

    def normalize(self, a, z):
        p_a, eu_a, d_a = a
        p_z, eu_z, d_z = z
        # each world I has p(I) and eu(I) = p(I) * u(I).
        # Normalizing the probability of the world (p(I)/p(Z) results p(I) / p(Z) * u(I) = eu(I) / p(Z)
        # Since total result = Sum_I [p(I) * u(I)]
        # print("normaling %s with %s" % (a,z))
        return p_a / p_z, eu_a / p_z, d_a

    def negate(self, a):
        return 1 - a[0], 0, a[2]

    def value(self, a):
        if type(a) is Constant:
            return float(a), 0, set()
        elif type(a) is Term and a.functor == '?':
            return 1, 0, set()
        elif type(a) is Term:
            return float(a.args[0]), (float(a.args[0]) * float(a.args[1])), a.args[2]
        else:
            raise ValueError("Could not interpret %s during conversion from external to internal representation." % a)

    def pos_value(self, a, key=None):
        if isinstance(a, pn_weight):
            return self.value(a.p_weight)
        else:
            return self.value(a)

    def neg_value(self, a, key=None):
        if isinstance(a, pn_weight):
            return self.value(a.n_weight)
        else:
            return self.negate(self.value(a))

    def is_dsp(self):
        return True

    def is_nsp(self):
        return True

    def in_domain(self, a):
        return 0.0 - 1e-9 <= a[0] <= 1.0 + 1e-9

    def to_evidence(self, pos_weight, neg_weight, sign):
        # Note: When eu = 0 because of p = 0 and now we set p = 1 then the eu can not be reconstructed and stays 0.
        p_p, p_eu, p_d = pos_weight
        n_p, n_eu, n_d = neg_weight

        if sign > 0:
            return pos_weight, (0.0, 0.0, n_d)
        else:
            return (0.0, 0.0, p_d), neg_weight

        # if sign > 0:
        #     if p_p == 0:  # normally does not happen because of inconsistentEvidenceError thrown before.
        #         return (1.0, 0.0, p_d), (0.0, 0.0, n_d)
        #     else:
        #         return (1.0, p_eu / p_p, p_d), (0.0, 0.0, n_d)  # positive weight rescaled to p=1
        # else:
        #     if n_p == 0:  # normally does not happen because of inconsistentEvidenceError thrown before.
        #         return (0.0, 0.0, p_d), (1.0, 0.0, n_d)
        #     else:
        #         return (0.0, 0.0, p_d), (1.0, n_eu / n_p, n_d)  # negative weight rescaled to p=1

    def ad_negate(self, pos_weight, neg_weight):
        p_p, p_eu, p_d = pos_weight
        n_p, n_eu, n_d = neg_weight

        neg_d = {-x for x in p_d}
        if n_p == 0:
            return 1.0, n_eu, neg_d  # TODO can this happen?
        else:
            return 1.0, n_eu / n_p, neg_d

    def ad_complement(self, ws, key=None):
        p, eu, d = ws[0]
        if len(d):
            return 0.0, 0.0, set()
        else: # normal p procedure + decision checks
            s = self.zero()
            for w in ws:
                if len(w[2]):
                    return None  # Trigger InvalidValueError. Must not mix decisions and probabilities #TODO Redo compliment check. Have a is_valid_ad method which returns a tuple of True/False and a message. Deprecate in_domain?
                s = self.plus(s, w)
            return self.negate(s)


class SemiringActiveUtilities(Semiring):
    """Semiring to collect all the active utilities.

    The purpose of this semiring is to collect into a set all the utilities that are not necessarily False.
    Each element can be None or a set of Terms.
    There's a limit on the number of active utilities retrieved that can be imposed. The idea is that if it is imposed
    then the decision search space gets smaller (because we will branch on a fixed number of children). The way this is
    implemented is: the merge function is the only place where sets increase, so if the merge creates a set bigger than
    the limit imposed, we select a subset of the merge (with size equal to the limit) with 20% of it which is taken from
    the utility parameters leading to the highest utility, and 80% of it which is taken randomly from the rest of the
    parameters.
    Notice that since we impose a constant limit, all the operations in the evaluation are then O(1), and not anymore
    proportional to the number of utility parameters.
    If the limit is not set, the circuit returns the whole set of active utility parameters.
    """

    def __init__(self, limit=None):
        self.emptyset = set()
        self._limit = limit
        self._rand_limit = math.ceil(limit * 0.8) if limit else limit

    def one(self):
        """Returns the identity element of the multiplication."""
        return self.emptyset

    def is_one(self, value):
        """Tests whether the given value is the identity element of the multiplication."""
        return (len(value) == 0) if value is not None else False

    def zero(self):
        """Returns the identity element of the addition."""
        return None

    def is_zero(self, value):
        """Tests whether the given value is the identity element of the addition."""
        return value is None

    def merge(self, a, b):
        merged = a | b
        if not self._limit or len(merged) <= self._limit:
            return merged
        else:
            det_limit = self._limit - self._rand_limit
            ordered_utils = sorted(list(merged), key=lambda x: x[1].compute_value(), reverse=True)
            new_list = ordered_utils[:det_limit]
            remaining_list = ordered_utils[det_limit:]
            if len(remaining_list) > 0:
                for _ in range(self._rand_limit):
                    new_list.append(remaining_list.pop(randint(0, len(remaining_list)-1)))

            return set(new_list)

    def plus(self, a, b):
        """Computes the addition of the given values."""
        if (a is not None) and (b is not None):
            # print(f"{a} + {b} = {a | b}")
            return self.merge(a, b)
        elif b is None:
            # print(f"{a} + {b} = {a}")
            return a
        elif a is None:
            # print(f"{a} + {b} = {b}")
            return b
        else:
            # print(f"{a} + {b} = None")
            return None

    def times(self, a, b):
        """Computes the multiplication of the given values."""
        if (a is not None) and (b is not None):
            # print(f"{a} x {b} = {a | b}")
            return self.merge(a,b)
        else:
            # print(f"{a} x {b} = None")
            return None

    def value(self, a):
        """Transform the given external value into an internal value."""
        if type(a) is Constant and float(a) == 0:
            # if weight is probability 0, then None is returned
            return None
        elif type(a) is Constant:
            # if weight is probability >0, we assign the emptyset
            return self.emptyset
        elif type(a) is Term and a.functor == '?':
            # if it is a decision we consider all possible assignments
            return self.emptyset
        elif isinstance(a, set) and len(a) == 1:
            return a
        else:
            raise ValueError("Could not interpret %s during conversion from external to internal representation." % a)

    def normalize(self, a, z):
        return a

    def pos_value(self, a, key=None):
        """Extract the positive internal value for the given external value."""
        return self.value(a)

    def neg_value(self, a, key=None):
        """Extract the negative internal value for the given external value."""
        if type(a) is Constant and float(a) == 1:
            # if weight is probability 1, negation has probability 0, thus None is returned
            return None
        elif type(a) is Constant:
            # if weight is probability <1, negation has probability >0, thus we assign the emptyset
            return self.emptyset
        elif type(a) is Term and a.functor == '?':
            # if it is a decision we consider all possible assignments
            return self.emptyset
        else:
            # negative value must be empty set otherwise we collect utilities from both sides
            assert(isinstance(a, set) and len(a) == 1)
            return self.emptyset

    def is_dsp(self):
        """Indicates whether this semiring requires solving a disjoint sum problem."""
        return True

    def is_nsp(self):
        """Indicates whether this semiring requires solving a neutral sum problem."""
        return True

    def in_domain(self, a):
        """Checks whether the given (internal) value is valid."""
        return isinstance(a, set) or a is None

    def ad_complement(self, ws, key=None):
        # This solves the case: 0.3::u1; 0.2::u2 :- action(stay).
        # That is, in an AD, when the probability is not summing up to one (i.e. it's not u1 nor u2).
        # In my case I want it to be the empty set.
        return self.emptyset

    def true(self, key=None):
        """Handle weight for deterministically true."""
        raise OperationNotSupported()

    def false(self, key=None):
        """Handle weight for deterministically false."""
        raise OperationNotSupported()

    def to_evidence(self, pos_weight, neg_weight, sign):
        """
        Converts the pos. and neg. weight (internal repr.) of a literal into the case where the literal is evidence.
        Note that the literal can be a negative atom regardless of the given sign.

        :param pos_weight: The current positive weight of the literal.
        :param neg_weight: The current negative weight of the literal.
        :param sign: Denotes whether the literal or its negation is evidence. sign > 0 denotes the literal is evidence,
            otherwise its negation is evidence. Note: The literal itself can also still be a negative atom.
        :returns: A tuple of the positive and negative weight as if the literal was evidence.
            For example, for probability, returns (self.one(), self.zero()) if sign else (self.zero(), self.one())
        """
        return (pos_weight, None) if sign else (None, neg_weight)

    def ad_negate(self, pos_weight, neg_weight):
        """
        Negation in the context of an annotated disjunction. e.g. in a probabilistic context for 0.2::a ; 0.8::b,
        the negative label for both a and b is 1.0 such that model {a,-b} = 0.2 * 1.0 and {-a,b} = 1.0 * 0.8.
        For a, pos_weight would be 0.2 and neg_weight could be 0.8. The returned value is 1.0.
        :param pos_weight: The current positive weight of the literal (e.g. 0.2 or 0.8). Internal representation.
        :param neg_weight: The current negative weight of the literal (e.g. 0.8 or 0.2). Internal representation.
        :return: neg_weight corrected based on the given pos_weight, given the ad context (e.g. 1.0). Internal
        representation.
        """
        return self.one()


class SemiringEnumModels(Semiring):
    def __init__(self):
        self.emptyset = set()

    def one(self):
        """Returns the identity element of the multiplication."""
        return 1.0, self.emptyset

    def is_one(self, value):
        """Tests whether the given value is the identity element of the multiplication."""
        return value == self.one

    def zero(self):
        """Returns the identity element of the addition."""
        return 0.0, self.emptyset

    def is_zero(self, value):
        """Tests whether the given value is the identity element of the addition."""
        return value == self.zero()

    def plus(self, a, b):
        """Computes the addition of the given values."""
        return a[0] + b[0], (a[1] | b[1])

    def times(self, a, b):
        """Computes the multiplication of the given values."""
        if a[0] == 0 or b[0] == 0:
            # print(f"{a} times {b} = 0, set()")
            return 0.0, self.emptyset
        else:
            if len(a[1]) == 0:
                new_set = b[1]
            elif len(b[1]) == 0:
                new_set = a[1]
            else:
                new_set = {frozenset(m1 | m2) for m1 in a[1] for m2 in b[1]}
                # print(f"{a} times {b} = ({a[0]*b[0]}, {new_set})")
            return a[0] * b[0], new_set

    def negate(self, a):
        """Returns the negation. This operation is optional.
        For example, for probabilities return 1-a.

        :raise OperationNotSupported: if the semiring does not support this operation
        """
        raise OperationNotSupported()

    def value(self, a: (float, Set[Term])):
        """Transform the given external value into an internal value.
        """
        return float(a[0]), a[1]

    def result(self, a, formula=None):
        """Transform the given internal value into an external value."""
        return a[0], a[1]

    def normalize(self, a, z):
        """Normalizes the given value with the given normalization constant.
        """
        return a[0]/z[0], a[1]

    def pos_value(self, a, key=None):
        """Extract the positive internal value for the given external value."""
        if isinstance(a, pn_weight):
            return a.p_weight
        elif type(a) == Term and a.functor == '?':
            return 0.5, {frozenset({key})}
        else:
            return float(a), {frozenset({key})}

    def neg_value(self, a, key=None):
        """Extract the negative internal value for the given external value."""
        if isinstance(a, pn_weight):
            return a.n_weight
        elif type(a) == Term and a.functor == '?':
            return 0.5, {frozenset({-key})}
        else:
            return 1-float(a), {frozenset({-key})}

    def result_zero(self):
        """Give the external representation of the identity element of the addition."""
        return self.result(self.zero())

    def result_one(self):
        """Give the external representation of the identity element of the multiplication."""
        return self.result(self.one())

    def is_dsp(self):
        """Indicates whether this semiring requires solving a disjoint sum problem."""
        return True

    def is_nsp(self):
        """Indicates whether this semiring requires solving a neutral sum problem."""
        return True

    def in_domain(self, a):
        """Checks whether the given (internal) value is valid."""
        return True

    def ad_complement(self, ws, key=None):
        s = self.zero()
        for w in ws:
            s = self.plus(s, w)
        return 1-s[0], self.emptyset

    def true(self, key=None):
        """Handle weight for deterministically true."""
        return self.one(), self.zero()

    def false(self, key=None):
        """Handle weight for deterministically false."""
        return self.zero(), self.one()

    def to_evidence(self, pos_weight, neg_weight, sign):
        """
        Converts the pos. and neg. weight (internal repr.) of a literal into the case where the literal is evidence.
        Note that the literal can be a negative atom regardless of the given sign.

        :param pos_weight: The current positive weight of the literal.
        :param neg_weight: The current negative weight of the literal.
        :param sign: Denotes whether the literal or its negation is evidence. sign > 0 denotes the literal is evidence,
            otherwise its negation is evidence. Note: The literal itself can also still be a negative atom.
        :returns: A tuple of the positive and negative weight as if the literal was evidence.
            For example, for probability, returns (self.one(), self.zero()) if sign else (self.zero(), self.one())
        """
        if sign > 0:
            return (1.0, pos_weight[1]), (0.0, neg_weight[1])
        else:
            return (0.0, pos_weight[1]), (1.0, neg_weight[1])

    def ad_negate(self, pos_weight, neg_weight):
        """
        Negation in the context of an annotated disjunction. e.g. in a probabilistic context for 0.2::a ; 0.8::b,
        the negative label for both a and b is 1.0 such that model {a,-b} = 0.2 * 1.0 and {-a,b} = 1.0 * 0.8.
        For a, pos_weight would be 0.2 and neg_weight could be 0.8. The returned value is 1.0.
        :param pos_weight: The current positive weight of the literal (e.g. 0.2 or 0.8). Internal representation.
        :param neg_weight: The current negative weight of the literal (e.g. 0.8 or 0.2). Internal representation.
        :return: neg_weight corrected based on the given pos_weight, given the ad context (e.g. 1.0). Internal
        representation.
        """
        return 1.0, neg_weight[1]
