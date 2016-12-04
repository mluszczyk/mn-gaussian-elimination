from unittest import TestCase

import numpy
from numpy.testing import assert_array_equal

from gauss import solvetopleft


class TestSolveTopLeft(TestCase):
    def test_one_element(self):
        result = solvetopleft(numpy.asmatrix([2.]), numpy.asmatrix([4.]))
        assert_array_equal(result, numpy.asmatrix([2.]))

    def test_id(self):
        m = numpy.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype='f')
        b = numpy.matrix([3, 1, 5], dtype='f').T
        r = solvetopleft(m, b)
        assert_array_equal(r, numpy.asmatrix([3, 1, 5]).T)

    def test_three_elements(self):
        m = numpy.matrix([[1, 4, 2], [0, 1, 3], [0, 0, 1]], dtype='f')
        b = numpy.array([-3, 11, 5], dtype='f')
        r = solvetopleft(m, b)
        assert_array_equal(r, numpy.array([3, -4, 5]))

    def test_division_on_diagonal(self):
        m = numpy.matrix([[2, 0], [0, 1]], dtype='f')
        b = numpy.array([4, 1], dtype='f')
        r = solvetopleft(m, b)
        assert_array_equal(r, [2, 1])

    def test_not_triangular_last_row(self):
        m = numpy.matrix([[2, 1], [1, 1]], dtype='f')
        b = numpy.array([1, 1], dtype='f')
        with self.assertRaises(Exception):
            solvetopleft(m, b)

    def test_not_triangular(self):
        m = numpy.matrix([[2, 1, 0], [1, 1, 0], [0, 0, 1]], dtype='f')
        b = numpy.array([1, 1, 1], dtype='f')
        with self.assertRaises(Exception):
            solvetopleft(m, b)
