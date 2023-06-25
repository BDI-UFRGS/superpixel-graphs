from setuptools import setup, Extension

import numpy

module1 = Extension('compute_features',
                    sources = ['compute_features.cpp'],
                    include_dirs=[numpy.get_include(), '/usr/local/include/opencv4', './include'],
                    library_dirs=['/usr/local/lib/'],
                    libraries=['opencv_core', 'opencv_imgcodecs', 'opencv_imgproc', 'opencv_ximgproc'])

setup (name = 'ComputeFeatures',
       version = '1.0',
       description = 'Computing .. features',
       ext_modules = [module1])