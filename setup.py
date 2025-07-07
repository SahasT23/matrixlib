from setuptools import setup, Extension
try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
    from pybind11 import get_include
    
    ext_modules = [
        Pybind11Extension(
            "matrix_ops",
            ["matrix_ops.cpp"],
            include_dirs=[get_include()],
            language='c++',
            cxx_std=11,
        ),
    ]
    
    setup(
        name="matrix_ops",
        ext_modules=ext_modules,
        cmdclass={"build_ext": build_ext},
        zip_safe=False,
        python_requires=">=3.6",
    )
    
except ImportError:
    # Fallback for older pybind11 versions
    import pybind11
    
    ext_modules = [
        Extension(
            "matrix_ops",
            ["matrix_ops.cpp"],
            include_dirs=[pybind11.get_include()],
            language='c++',
            extra_compile_args=['-std=c++11'],
        ),
    ]
    
    setup(
        name="matrix_ops",
        ext_modules=ext_modules,
        zip_safe=False,
        python_requires=">=3.6",
    )