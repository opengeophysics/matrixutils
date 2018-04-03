import numpy as np


def meshTensor(value):
    """**meshTensor** takes a list of numbers and tuples
    that have the form::

        mT = [ float, (cellSize, numCell), (cellSize, numCell, factor) ]

    For example, a time domain mesh code needs
    many time steps at one time::

        [(1e-5, 30), (1e-4, 30), 1e-3]

    Means take 30 steps at 1e-5 and then 30 more at 1e-4,
    and then one step of 1e-3.

    Tensor meshes can also be created by increase factors::

        [(10.0, 5, -1.3), (10.0, 50), (10.0, 5, 1.3)]

    When there is a third number in the tuple, it
    refers to the increase factor, if this number
    is negative this section of the tensor is flipped right-to-left.

    """
    if type(value) is not list:
        raise Exception('meshTensor must be a list of scalars and tuples.')

    proposed = []
    for v in value:
        if np.isscalar(v):
            proposed += [float(v)]
        elif type(v) is tuple and len(v) == 2:
            proposed += [float(v[0])]*int(v[1])
        elif type(v) is tuple and len(v) == 3:
            start = float(v[0])
            num = int(v[1])
            factor = float(v[2])
            pad = ((np.ones(num)*np.abs(factor))**(np.arange(num)+1))*start
            if factor < 0: pad = pad[::-1]
            proposed += pad.tolist()
        else:
            raise Exception('meshTensor must contain only scalars and len(2) or len(3) tuples.')

    return np.array(proposed)
