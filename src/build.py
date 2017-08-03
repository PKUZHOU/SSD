# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = "bbious",
    ext_modules = cythonize("bbious.pyx")
)