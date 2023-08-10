"""
Definition of the semirings used by the tool.
"""

from collections import namedtuple

import numpy as np
import tensorflow as tf

Label = namedtuple('Label', 'prob, eu')

class EUSemiring:
    def __init__(self):
        self.val_zero = (0.0, 0.0)
        self.val_one = (1.0, 0.0)

    def one(self):
        return self.val_one

    def zero(self):
        return self.val_zero

    def plus(self, a, b):
        p_a, eu_a = a
        p_b, eu_b = b
        # print("(%s, %s) + (%s, %s) = (%s, %s)" % (p_a, eu_a, p_b, eu_b, p_a + p_b, eu_a + eu_b))
        return p_a + p_b, eu_a + eu_b

    @staticmethod
    def times(a, b):
        p_a, eu_a = a
        p_b, eu_b = b
        # print("(%s, %s) * (%s, %s) = (%s, %s)" % (p_a, eu_a, p_b, eu_b, p_a * p_b, p_a * eu_b + p_b * eu_a))
        return p_a * p_b, p_a * eu_b + p_b * eu_a

    @staticmethod
    def value(a):
        return a[0], a[1]

    @staticmethod
    def normalise(a, z):
        p_a, eu_a = a
        p_z, _ = z
        return p_a / p_z, eu_a / p_z

class MEUSemiring:
    def __init__(self):
        self.val_zero = (0.0, 0.0, True)
        self.val_one = (1.0, 0.0, False)

    def one(self):
        return self.val_one

    def zero(self):
        return self.val_zero

    def plus(self, a, b):
        p_a, eu_a, max_a = a
        p_b, eu_b, max_b = b
        if max_a or max_b:
            take_a = (p_b == 0) | ((p_a != 0) & (eu_a / p_a > eu_b / p_b))
            p_res = tf.where(take_a, p_a, p_b)
            eu_res = tf.where(take_a, eu_a, eu_b)
            # def return_a(): return a
            # def return_b(): return b
            # result = tf.case([(p_a == 0, return_b), (p_b == 0, return_a), (eu_a / p_a >= eu_b / p_b, return_a)],
            #                  default=return_b, exclusive=True)
            # print("max( (%s, %s, %s), (%s, %s, %s) ) = (%s, %s, %s)" %
            #       (a.prob, a.eu, a.dec, b.prob, b.eu, b.dec, result.prob, result.eu, result.dec))
            return p_res, eu_res, True
        else:
            # print("(%s, %s) + (%s, %s) = (%s, %s)" %
            #       (a.prob, a.eu, b.prob, b.eu, p_a + p_b, eu_a + eu_b))
            return p_a + p_b, eu_a + eu_b, False

    @staticmethod
    def times(a, b):
        p_a, eu_a, max_a = a
        p_b, eu_b, max_b = b
        eu_n = p_a * eu_b + p_b * eu_a
        # print("(%s, %s) * (%s, %s) = (%s, %s)" %
        #       (p_a, eu_a, p_b, eu_b, p_a * p_b, eu_n))
        return p_a * p_b, eu_n, max_a or max_b

    @staticmethod
    def value(a):
        # Max: since all the decisions are on top (X-constrained SDDs), as long as there is some decisions, it means
        # we have to maximise. Because of the smoothness of the circuit we know the two sets are different.
        return a.prob, a.eu, len(a.dec) > 0

    @staticmethod
    def normalise(a, z):
        p_a, eu_a, max_a = a
        p_z, _, _ = z
        return p_a / p_z, eu_a / p_z, max_a


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
