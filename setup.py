from setuptools import setup, Extension

import numpy

module1 = Extension(name='superpixel_graphs.graph_builder',
                    sources = ['superpixel_graphs/graph_builder.cpp'],
                    include_dirs=[numpy.get_include(), '/usr/local/include/opencv4', 'superpixel_graphs'],
                    library_dirs=['/usr/local/lib/'],
                    libraries=['opencv_core', 'opencv_imgcodecs', 'opencv_imgproc', 'opencv_ximgproc'])

setup (ext_modules = [module1])