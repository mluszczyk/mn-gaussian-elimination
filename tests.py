from unittest import TestCase

import numpy
from numpy.testing import assert_array_equal

from gauss import solveupper, triangulate, solve


class TestSolveUpper(TestCase):
    def test_one_element(self):
        result = solveupper(numpy.asarray([[2.]]), numpy.asarray([4.]))
        assert_array_equal(result, numpy.asarray([2.]))

    def test_id(self):
        m = numpy.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype='f')
        b = numpy.array([3, 1, 5], dtype='f').T
        r = solveupper(m, b)
        assert_array_equal(r, numpy.asarray([3, 1, 5]).T)

    def test_three_elements(self):
        m = numpy.array([[1, 4, 2], [0, 1, 3], [0, 0, 1]], dtype='f')
        b = numpy.array([-3, 11, 5], dtype='f')
        r = solveupper(m, b)
        assert_array_equal(r, numpy.array([3, -4, 5]))

    def test_division_on_diagonal(self):
        m = numpy.array([[2, 0], [0, 1]], dtype='f')
        b = numpy.array([4, 1], dtype='f')
        r = solveupper(m, b)
        assert_array_equal(r, [2, 1])

    def test_not_triangular_last_row(self):
        m = numpy.array([[2, 1], [1, 1]], dtype='f')
        b = numpy.array([1, 1], dtype='f')
        with self.assertRaises(Exception):
            solveupper(m, b)

    def test_not_triangular(self):
        m = numpy.array([[2, 1, 0], [1, 1, 0], [0, 0, 1]], dtype='f')
        b = numpy.array([1, 1, 1], dtype='f')
        with self.assertRaises(Exception):
            solveupper(m, b)


class TestTriangulate(TestCase):
    def test_one_element(self):
        r = numpy.array([[2., 2.]], dtype='f')
        triangulate(r)
        assert_array_equal(r, numpy.array([[1., 1.]], dtype='f'))

    def test_two_elements(self):
        m = numpy.array([[2., 2., 4.], [1., 2., 0.]], dtype='f')
        triangulate(m)
        expected = numpy.array([[1., 1., 2.], [0., 1., -2]], dtype='f')
        assert_array_equal(m, expected)

    def test_nonsingular(self):
        m = numpy.array([[1., 2., 1.], [2., 4., 2.]])
        with self.assertRaises(Exception):
            triangulate(m)

    def test_element_choice(self):
        m = numpy.array([[2., 4.], [3., 9.]])
        triangulate(m)
        expected = numpy.array([[1., 3.], [0., 1.]])
        assert_array_equal(m, expected)


class TestGauss(TestCase):
    def test(self):
        m = numpy.array([[1., 2., 3.], [0., 1., 2.], [0., 0., 1.]])
        b = numpy.array([10., 4., 1.])
        result = solve(m, b)
        expected = numpy.array([3., 2., 1.])
        assert_array_equal(result, expected)
