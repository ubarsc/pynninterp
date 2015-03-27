from distutils.core import setup, Extension

pynninterp = Extension('pynninterp',
                sources = ['delaunay.c', 'delaunay.h', 'hash.c', 'hash.h', 'istack.c', 'istack.h', 'lpi.c', 'minell.c', 'minell.h', 'nan.h', 'nn.h', 'nn_internal.h', 'nnai.c', 'nncommon-vulnerable.c', 'nncommon.c', 'nnpi.c', 'preader.c', 'preader.h', 'triangle.c', 'triangle.h', 'version.h'])

setup (name = 'PyNNInterp',
       version = '1.0',
       description = 'This is a package for running a natural neighbour interpolation from python.',
       ext_modules = [pynninterp])