#!/usr/bin/env python
from __future__ import print_function
"""matrixutils

Utilities for working with matrices as linear operators
"""

from setuptools import find_packages

try:
    from numpy.distutils.core import setup
except Exception:
    raise Exception(
        "Install requires numpy. "
        "If you use conda, `conda install numpy` "
        "or you can use pip, `pip install numpy`"
    )


CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Physics',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Operating System :: MacOS',
    'Natural Language :: English',
]

with open("README.rst") as f:
    LONG_DESCRIPTION = ''.join(f.readlines())


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('matrixutils')

    return config


setup(
    name="matrixutils",
    version="0.0.2",
    install_requires=[
        'numpy>=1.7',
        'scipy>=0.13',
        'cython',
        'matplotlib',
        'properties>=0.3.6b0',
    ],
    author="Open Geophysics Developers",
    author_email="admin@simpeg.xyz",
    description="Utilities for working with matrices as linear operators",
    long_description=LONG_DESCRIPTION,
    license="MIT",
    keywords="finite volume, discretization, pde, ode",
    url="http://simpeg.xyz/",
    download_url="https://github.com/opengeophysics/matrixutils",
    classifiers=CLASSIFIERS,
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    use_2to3=False,
    setup_requires=['numpy'],
    configuration=configuration
)
