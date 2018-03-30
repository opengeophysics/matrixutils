from __future__ import print_function

from .matutils import (
    mkvc, sdiag, sdInv, speye, kron3, spzeros, ddx, av,
    av_extrap, ndgrid, ind2sub, sub2ind, getSubArray,
    inv3X3BlockDiagonal, inv2X2BlockDiagonal,
    Zero, Identity
)
from .codeutils import (isScalar, asArray_N_x_Dim)
from .meshutils import meshTensor
from .curvutils import volTetra, faceInfo, indexCube
from .interputils import interpmat
from .coordutils import rotatePointsFromNormals, rotationMatrixFromNormals


__version__   = '0.0.1'
__author__    = 'OpenGeophysics Team'
__license__   = 'MIT'
__copyright__ = '2013 - 2018, OpenGeophysics Developers, http://simpeg.xyz'
