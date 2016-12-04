import numpy


def solvetopleft(mat, b):
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

