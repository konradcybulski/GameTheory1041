try:
    from distutils.core import setup
except ImportError:
    from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("simulation_instance.pyx"),
    package_dir={'Cython_resources': ''},
    include_dirs=[numpy.get_include()],
)