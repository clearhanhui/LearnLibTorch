from setuptools import Extension, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, include_paths, library_paths


# libraries = ['c10', 'torch', 'torch_cpu', 'torch_python']
# modules = [Extension(
#    name='gc_cpp', # package name
#    sources=['gc_layer.cpp'],
#    include_dirs=include_paths(),
#    library_dirs=library_paths(),
#    libraries=libraries,
#    language='c++')]


# The next line is equivalent to the above code snippets
modules = [CppExtension('gc_cpp', ['gc_layer.cpp'])]

setup(
    name='gc_cpp', # pypi name, not package name.
    version='0.0.1',
    description='GCN layer forward and backward extension of pytorch',
    author='clearhanhui',
    author_email='clearhanhui@gmail.com',

    ext_modules=modules,
    cmdclass={'build_ext': BuildExtension}
)
