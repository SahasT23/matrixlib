from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'matrixlib.matrix',
        ['src/matrix.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['-std=c++11'],
    ),
]

setup(
    name='matrixlib',
    version='0.1.0',
    description='A matrix and vector computation library with C++ backend',
    ext_modules=ext_modules,
    packages=['matrixlib'],
    install_requires=['pybind11>=2.6'],
)