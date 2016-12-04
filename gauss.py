import numpy


def solvetopleft(mat, b):
    assert mat.ndim == 2
    assert b.ndim == 1
    assert mat.shape[0] == mat.shape[1]
    assert b.shape[0] == mat.shape[0]
    n = mat.shape[0]
    assert (mat[n - 1, :n - 1] == 0).all(), "nonsingular matrix!"
    res = b.copy()
    res[n - 1] /= mat[n - 1, n - 1]
    for c in range(n - 2, -1, -1):
        assert mat[c, c] != 0, "nonsingular matrix!"
        assert (mat[c, :c] == 0).all(), "not a triangular matrix"
        res[c] = (b[c] - numpy.inner(mat[c, c + 1:].reshape(-1), res[c + 1:].reshape(-1))) / mat[c, c]
    return res


def triangulate(mat):
    """Triangulizes a nonsingular matrix in place."""
    assert mat.shape[0] <= mat.shape[1]
    assert mat.ndim == 2

    n = mat.shape[0]

    for c in range(n):
        assert (mat[c, :c] == 0).all(), "not elimanted correctly"
        assert (mat[c, c] != 0).all(), "nonsingular matrix"
        row = numpy.argmax(mat[c:, c]) + c
        mat[[row, c]] = mat[[c, row]]
        mat[c] /= mat[c, c]
        for d in range(c + 1, n):
            mat[d] -= mat[c] * mat[d, c]


def solve(mat, b):
    assert mat.ndim == 2
    assert b.ndim == 1
    assert mat.shape[0] == mat.shape[1]
    assert b.shape[0] == mat.shape[0]
    n = mat.shape[0]
    b = b.reshape(-1, 1)
    full = numpy.concatenate((mat, b), axis=1)
    triangulate(full)
    mat, b = full[:, :n], full[:, n:]
    b = b.flatten()
    return solvetopleft(mat, b)
