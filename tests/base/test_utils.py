from __future__ import print_function
import unittest
import numpy as np
import scipy.sparse as sp
from discretize.utils import (
    sdiag, sub2ind, ndgrid, mkvc, isScalar,
    inv2X2BlockDiagonal, inv3X3BlockDiagonal,
    invPropertyTensor, makePropertyTensor, indexCube,
    ind2sub, asArray_N_x_Dim, TensorType, Zero, Identity
)
from discretize.Tests import checkDerivative
import discretize
import sys

TOL = 1e-8


class TestCheckDerivative(unittest.TestCase):

    def test_simplePass(self):
        def simplePass(x):
            return np.sin(x), sdiag(np.cos(x))
        passed = checkDerivative(simplePass, np.random.randn(5), plotIt=False)
        self.assertTrue(passed, True)

    def test_simpleFunction(self):
        def simpleFunction(x):
            return np.sin(x), lambda xi: sdiag(np.cos(x))*xi
        passed = checkDerivative(simpleFunction, np.random.randn(5), plotIt=False)
        self.assertTrue(passed, True)

    def test_simpleFail(self):
        def simpleFail(x):
            return np.sin(x), -sdiag(np.cos(x))
        passed = checkDerivative(simpleFail, np.random.randn(5), plotIt=False)
        self.assertTrue(not passed, True)


class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        self.a = np.array([1, 2, 3])
        self.b = np.array([1, 2])
        self.c = np.array([1, 2, 3, 4])

    def test_mkvc1(self):
        x = mkvc(self.a)
        self.assertTrue(x.shape, (3, ))

    def test_mkvc2(self):
        x = mkvc(self.a, 2)
        self.assertTrue(x.shape, (3, 1))

    def test_mkvc3(self):
        x = mkvc(self.a, 3)
        self.assertTrue(x.shape, (3, 1, 1))

    def test_ndgrid_2D(self):
        XY = ndgrid([self.a, self.b])

        X1_test = np.array([1, 2, 3, 1, 2, 3])
        X2_test = np.array([1, 1, 1, 2, 2, 2])

        self.assertTrue(np.all(XY[:, 0] == X1_test))
        self.assertTrue(np.all(XY[:, 1] == X2_test))

    def test_ndgrid_3D(self):
        XYZ = ndgrid([self.a, self.b, self.c])

        X1_test = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])
        X2_test = np.array([1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2])
        X3_test = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4])

        self.assertTrue(np.all(XYZ[:, 0] == X1_test))
        self.assertTrue(np.all(XYZ[:, 1] == X2_test))
        self.assertTrue(np.all(XYZ[:, 2] == X3_test))

    def test_sub2ind(self):
        x = np.ones((5, 2))
        self.assertTrue(np.all(sub2ind(x.shape, [0, 0]) == [0]))
        self.assertTrue(np.all(sub2ind(x.shape, [4, 0]) == [4]))
        self.assertTrue(np.all(sub2ind(x.shape, [0, 1]) == [5]))
        self.assertTrue(np.all(sub2ind(x.shape, [4, 1]) == [9]))
        self.assertTrue(np.all(sub2ind(x.shape, [[4, 1]]) == [9]))
        self.assertTrue(np.all(sub2ind(x.shape, [[0, 0], [4, 0], [0, 1], [4, 1]]) == [0, 4, 5, 9]))

    def test_ind2sub(self):
        x = np.ones((5, 2))
        self.assertTrue(np.all(ind2sub(x.shape, [0, 4, 5, 9])[0] == [0, 4, 0, 4]))
        self.assertTrue(np.all(ind2sub(x.shape, [0, 4, 5, 9])[1] == [0, 0, 1, 1]))

    def test_indexCube_2D(self):
        nN = np.array([3, 3])
        self.assertTrue(np.all(indexCube('A', nN) == np.array([0, 1, 3, 4])))
        self.assertTrue(np.all(indexCube('B', nN) == np.array([3, 4, 6, 7])))
        self.assertTrue(np.all(indexCube('C', nN) == np.array([4, 5, 7, 8])))
        self.assertTrue(np.all(indexCube('D', nN) == np.array([1, 2, 4, 5])))

    def test_indexCube_3D(self):
        nN = np.array([3, 3, 3])
        self.assertTrue(np.all(indexCube('A', nN) == np.array([0, 1, 3, 4, 9, 10, 12, 13])))
        self.assertTrue(np.all(indexCube('B', nN) == np.array([3, 4, 6, 7, 12, 13, 15, 16])))
        self.assertTrue(np.all(indexCube('C', nN) == np.array([4, 5, 7, 8, 13, 14, 16, 17])))
        self.assertTrue(np.all(indexCube('D', nN) == np.array([1, 2, 4, 5, 10, 11, 13, 14])))
        self.assertTrue(np.all(indexCube('E', nN) == np.array([9, 10, 12, 13, 18, 19, 21, 22])))
        self.assertTrue(np.all(indexCube('F', nN) == np.array([12, 13, 15, 16, 21, 22, 24, 25])))
        self.assertTrue(np.all(indexCube('G', nN) == np.array([13, 14, 16, 17, 22, 23, 25, 26])))
        self.assertTrue(np.all(indexCube('H', nN) == np.array([10, 11, 13, 14, 19, 20, 22, 23])))

    def test_invXXXBlockDiagonal(self):
        a = [np.random.rand(5, 1) for i in range(4)]

        B = inv2X2BlockDiagonal(*a)

        A = sp.vstack((sp.hstack((sdiag(a[0]), sdiag(a[1]))),
                       sp.hstack((sdiag(a[2]), sdiag(a[3])))))

        Z2 = B*A - sp.identity(10)
        self.assertTrue(np.linalg.norm(Z2.todense().ravel(), 2) < TOL)

        a = [np.random.rand(5, 1) for i in range(9)]
        B = inv3X3BlockDiagonal(*a)

        A = sp.vstack((sp.hstack((sdiag(a[0]), sdiag(a[1]),  sdiag(a[2]))),
                       sp.hstack((sdiag(a[3]), sdiag(a[4]),  sdiag(a[5]))),
                       sp.hstack((sdiag(a[6]), sdiag(a[7]),  sdiag(a[8])))))

        Z3 = B*A - sp.identity(15)

        self.assertTrue(np.linalg.norm(Z3.todense().ravel(), 2) < TOL)

    def test_invPropertyTensor2D(self):
        M = discretize.TensorMesh([6, 6])
        a1 = np.random.rand(M.nC)
        a2 = np.random.rand(M.nC)
        a3 = np.random.rand(M.nC)
        prop1 = a1
        prop2 = np.c_[a1, a2]
        prop3 = np.c_[a1, a2, a3]

        for prop in [4, prop1, prop2, prop3]:
            b = invPropertyTensor(M, prop)
            A = makePropertyTensor(M, prop)
            B1 = makePropertyTensor(M, b)
            B2 = invPropertyTensor(M, prop, returnMatrix=True)

            Z = B1*A - sp.identity(M.nC*2)
            self.assertTrue(np.linalg.norm(Z.todense().ravel(), 2) < TOL)
            Z = B2*A - sp.identity(M.nC*2)
            self.assertTrue(np.linalg.norm(Z.todense().ravel(), 2) < TOL)

    def test_TensorType2D(self):
        M = discretize.TensorMesh([6, 6])
        a1 = np.random.rand(M.nC)
        a2 = np.random.rand(M.nC)
        a3 = np.random.rand(M.nC)
        prop1 = a1
        prop2 = np.c_[a1, a2]
        prop3 = np.c_[a1, a2, a3]

        for ii, prop in enumerate([4, prop1, prop2, prop3]):
            self.assertTrue(TensorType(M, prop) == ii)

        self.assertRaises(Exception, TensorType, M, np.c_[a1, a2, a3, a3])
        self.assertTrue(TensorType(M, None) == -1)

    def test_TensorType3D(self):
        M = discretize.TensorMesh([6, 6, 7])
        a1 = np.random.rand(M.nC)
        a2 = np.random.rand(M.nC)
        a3 = np.random.rand(M.nC)
        a4 = np.random.rand(M.nC)
        a5 = np.random.rand(M.nC)
        a6 = np.random.rand(M.nC)
        prop1 = a1
        prop2 = np.c_[a1, a2, a3]
        prop3 = np.c_[a1, a2, a3, a4, a5, a6]

        for ii, prop in enumerate([4, prop1, prop2, prop3]):
            self.assertTrue(TensorType(M, prop) == ii)

        self.assertRaises(Exception, TensorType, M, np.c_[a1, a2, a3, a3])
        self.assertTrue(TensorType(M, None) == -1)

    def test_invPropertyTensor3D(self):
        M = discretize.TensorMesh([6, 6, 6])
        a1 = np.random.rand(M.nC)
        a2 = np.random.rand(M.nC)
        a3 = np.random.rand(M.nC)
        a4 = np.random.rand(M.nC)
        a5 = np.random.rand(M.nC)
        a6 = np.random.rand(M.nC)
        prop1 = a1
        prop2 = np.c_[a1, a2, a3]
        prop3 = np.c_[a1, a2, a3, a4, a5, a6]

        for prop in [4, prop1, prop2, prop3]:
            b = invPropertyTensor(M, prop)
            A = makePropertyTensor(M, prop)
            B1 = makePropertyTensor(M, b)
            B2 = invPropertyTensor(M, prop, returnMatrix=True)

            Z = B1*A - sp.identity(M.nC*3)
            self.assertTrue(np.linalg.norm(Z.todense().ravel(), 2) < TOL)
            Z = B2*A - sp.identity(M.nC*3)
            self.assertTrue(np.linalg.norm(Z.todense().ravel(), 2) < TOL)

    def test_isScalar(self):
        self.assertTrue(isScalar(1.))
        self.assertTrue(isScalar(1))
        if sys.version_info < (3, ):
            self.assertTrue(isScalar(long(1)))
        self.assertTrue(isScalar(np.r_[1.]))
        self.assertTrue(isScalar(np.r_[1]))

    def test_asArray_N_x_Dim(self):

        true = np.array([[1, 2, 3]])

        listArray = asArray_N_x_Dim([1, 2, 3], 3)
        self.assertTrue(np.all(true == listArray))
        self.assertTrue(true.shape == listArray.shape)

        listArray = asArray_N_x_Dim(np.r_[1, 2, 3], 3)
        self.assertTrue(np.all(true == listArray))
        self.assertTrue(true.shape == listArray.shape)

        listArray = asArray_N_x_Dim(np.array([[1, 2, 3.]]), 3)
        self.assertTrue(np.all(true == listArray))
        self.assertTrue(true.shape == listArray.shape)

        true = np.array([[1, 2], [4, 5]])

        listArray = asArray_N_x_Dim([[1, 2], [4, 5]], 2)
        self.assertTrue(np.all(true == listArray))
        self.assertTrue(true.shape == listArray.shape)


class TestZero(unittest.TestCase):

    def test_zero(self):
        z = Zero()
        assert z == 0
        assert not (z < 0)
        assert z <= 0
        assert not (z > 0)
        assert z >= 0
        assert +z == z
        assert -z == z
        assert z + 1 == 1
        assert z + 3 + z == 3
        assert z - 3 == -3
        assert z - 3 - z == -3
        assert 3*z == 0
        assert z*3 == 0
        assert z/3 == 0

        a = 1
        a += z
        assert a == 1
        a = 1
        a += z
        assert a == 1
        self.assertRaises(ZeroDivisionError, lambda: 3/z)

        assert mkvc(z) == 0
        assert sdiag(z)*a == 0
        assert z.T == 0
        assert z.transpose() == 0

    def test_mat_zero(self):
        z = Zero()
        S = sdiag(np.r_[2, 3])
        assert S*z == 0

    def test_numpy_multiply(self):
        z = Zero()
        x = np.r_[1, 2, 3]
        a = x * z
        assert isinstance(a, Zero)

        z = Zero()
        x = np.r_[1, 2, 3]
        a = z * x
        assert isinstance(a, Zero)

    def test_one(self):
        o = Identity()
        assert o == 1
        assert not (o < 1)
        assert o <= 1
        assert not (o > 1)
        assert o >= 1
        o = -o
        assert o == -1
        assert not (o < -1)
        assert o <= -1
        assert not (o > -1)
        assert o >= -1
        assert -1.*(-o)*o == -o
        o = Identity()
        assert +o == o
        assert -o == -o
        assert o*3 == 3
        assert -o*3 == -3
        assert -o*o == -1
        assert -o*o*-o == 1
        assert -o + 3 == 2
        assert 3 + -o == 2

        assert -o - 3 == -4
        assert o - 3 == -2
        assert 3 - -o == 4
        assert 3 - o == 2

        assert o//2 == 0
        assert o/2. == 0.5
        assert -o//2 == -1
        assert -o/2. == -0.5
        assert 2/o == 2
        assert 2/-o == -2

        assert o.T == 1
        assert o.transpose() == 1

    def test_mat_one(self):

        o = Identity()
        S = sdiag(np.r_[2, 3])

        def check(exp, ans):
            assert np.all((exp).todense() == ans)

        check(S * o, [[2, 0], [0, 3]])
        check(o * S, [[2, 0], [0, 3]])
        check(S * -o, [[-2, 0], [0, -3]])
        check(-o * S, [[-2, 0], [0, -3]])
        check(S/o, [[2, 0], [0, 3]])
        check(S/-o, [[-2, 0], [0, -3]])
        self.assertRaises(NotImplementedError, lambda: o/S)

        check(S + o, [[3, 0], [0, 4]])
        check(o + S, [[3, 0], [0, 4]])
        check(S - o, [[1, 0], [0, 2]])

        check(S + - o, [[1, 0], [0, 2]])
        check(- o + S, [[1, 0], [0, 2]])

    def test_mat_shape(self):
        o = Identity()
        S = sdiag(np.r_[2, 3])[:1, :]
        self.assertRaises(ValueError, lambda: S + o)

        def check(exp, ans):
            assert np.all((exp).todense() == ans)

        check(S * o, [[2, 0]])
        check(S * -o, [[-2, 0]])

    def test_numpy_one(self):
        o = Identity()
        n = np.r_[2., 3]

        assert np.all(n+1 == n+o)
        assert np.all(1+n == o+n)
        assert np.all(n-1 == n-o)
        assert np.all(1-n == o-n)
        assert np.all(n/1 == n/o)
        assert np.all(n/-1 == n/-o)
        assert np.all(1/n == o/n)
        assert np.all(-1/n == -o/n)
        assert np.all(n*1 == n*o)
        assert np.all(n*-1 == n*-o)
        assert np.all(1*n == o*n)
        assert np.all(-1*n == -o*n)

    def test_both(self):
        z = Zero()
        o = Identity()
        assert o*z == 0
        assert o*z + o == 1
        assert o-z == 1

if __name__ == '__main__':
    unittest.main()