from distutils.core import setup, Extension

pynninterp = Extension('pynninterp',
                define_macros = [('MAJOR_VERSION', '1'),
                                 ('MINOR_VERSION', '0')],
                sources = ['src/delaunay.c', 'src/hash.c', 'src/istack.c', 'src/lpi.c', 'src/minell.c', 'src/nnai.c', 'src/nncommon-vulnerable.c', 'src/nncommon.c', 'src/nnpi.c', 'src/preader.c', 'src/triangle.c'])

setup (name = 'PyNNInterp',
       version = '1.0',
       description = 'This is a package for running a natural neighbour interpolation from python.',
       author = 'John Armston, Pete Bunting, Neil Flood, Sam Gillingham',
       author_email = 'petebunting@mac.com',
       ext_modules = [pynninterp])