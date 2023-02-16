"""
Definition of the semirings used by the tool.
"""

from collections import namedtuple

import numpy as np
import torch

Label = namedtuple('Label', 'prob, eu, dec')


class MEUSemiring:
    MEULabel = namedtuple('MEULabel', 'prob, eu, max')

    def __init__(self):
        self.val_zero = MEUSemiring.MEULabel(0.0, 0.0, True)
        self.val_one = MEUSemiring.MEULabel(1.0, 0.0, False)

    def one(self):
        return self.val_one

    def zero(self):
        return self.val_zero

    def plus(self, a: MEULabel, b: MEULabel) -> MEULabel:
        p_a, eu_a, max_a = a
        p_b, eu_b, max_b = b
        if max_a or max_b:
            take_a = (p_b == 0) | ((p_a != 0) & (eu_a / p_a > eu_b / p_b))
            p_res = (p_a * take_a) + (p_b * torch.logical_not(take_a))
            eu_res = (eu_a * take_a) + (eu_b * torch.logical_not(take_a))
            # print("max( (%s, %s), (%s, %s) ) = (%s, %s)" %
            #       (a.prob, a.eu, b.prob, b.eu, p_res, eu_res))
            return MEUSemiring.MEULabel(p_res, eu_res, True)
        else:
            # print("(%s, %s) + (%s, %s) = (%s, %s)" %
            #       (a.prob, a.eu, b.prob, b.eu, p_a + p_b, eu_a + eu_b))
            return MEUSemiring.MEULabel(p_a + p_b, eu_a + eu_b, False)

    @staticmethod
    def times(a: MEULabel, b: MEULabel) -> MEULabel:
        p_a, eu_a, max_a = a
        p_b, eu_b, max_b = b
        eu_n = p_a * eu_b + p_b * eu_a
        # print("(%s, %s) * (%s, %s) = (%s, %s)" %
        #       (a.prob, a.eu, b.prob, b.eu, p_a * p_b, eu_n))
        return MEUSemiring.MEULabel(p_a * p_b, eu_n, max_a or max_b)

    @staticmethod
    def value(a: Label) -> MEULabel:
        # Max: since all the decisions are on top (X-constrained SDDs), as long as there is some decisions, it means
        # we have to maximise. Because of the smoothness of the circuit we know the two sets are different.
        return MEUSemiring.MEULabel(a.prob, a.eu, len(a.dec) > 0)

    @staticmethod
    def normalise(a: MEULabel, z: MEULabel) -> MEULabel:
        return MEUSemiring.MEULabel(a.prob / z.prob, a.eu / z.prob, a.max)


class BestDecSemiring:
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
            # Max: since all the decisions are on top (X-constrained SDDs), as long as there is some decisions, it means
            # we have to maximise. Because of the smoothness of the circuit we know the two sets are different.
            # This is more efficient than checking if the two sets are different.
            if len(d_b) == self.zero_decision_length:
                result = a
            elif len(d_a) == self.zero_decision_length:
                result = b
            elif p_a == 0:
                result = b
            elif p_b == 0:
                result = a
            elif eu_a / p_a >= eu_b / p_b:  # check on eu
                result = a
            else:
                result = b
            # print("max( (%s, %s, %s), (%s, %s, %s) ) = (%s, %s, %s)" %
            #       (a.prob, a.eu, a.dec, b.prob, b.eu, b.dec, result.prob, result.eu, result.dec))
            return result
        else:
            # Sum
            # print("(%s, %s, %s) + (%s, %s, %s) = (%s, %s, %s)" %
            #       (a.prob, a.eu, a.dec, b.prob, b.eu, b.dec, p_a + p_b, eu_a + eu_b, self._empty_set))
            return Label(p_a + p_b, eu_a + eu_b, self._empty_set)

    def times(self, a: Label, b: Label):
        p_a, eu_a, d_a = a
        p_b, eu_b, d_b = b
        eu_n = p_a * eu_b + p_b * eu_a
        # print("(%s, %s, %s) * (%s, %s, %s) = (%s, %s, %s)" %
        #       (a.prob, a.eu, a.dec, b.prob, b.eu, b.dec, p_a * p_b, eu_n, d_a.union(d_b)))
        d = self._empty_set
        if len(d_a) == 0:
            d = d_b
        elif len(d_b) == 0:
            d = d_a
        elif len(d_a) != 0 and len(d_b) != 0:
            d = d_a.union(d_b)
        return Label(p_a * p_b, eu_n, d)

    # @staticmethod
    # def value(a: Label) -> Label:
    #     return a

    @staticmethod
    def normalise(a: Label, z: Label) -> Label:
        return Label(a.prob / z.prob, a.eu / z.prob, a.dec)
