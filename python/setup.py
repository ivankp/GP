from distutils.core import setup, Extension
import sysconfig

setup(
    name = 'ivanp_gp',
    author = 'Ivan Pogrebnyak',
    url='https://github.com/ivankp/GP',
    ext_modules = [
    Extension('ivanp_gp',
        sources = ['gp.cc','../src/linalg.cc'],
        libraries=['gsl','gslcblas'],
        include_dirs = ['../include'],
        language = 'c++14',
        extra_compile_args = sysconfig.get_config_var('CFLAGS').split() +
            ['-std=c++14','-Wall','-O3'],
        extra_link_args = ['-lstdc++']
    )
])
