from distutils.core import setup, Extension
import sysconfig

setup(
    name = 'gaussian_process',
    author = 'Ivan Pogrebnyak',
    url='https://github.com/ivankp/hgam_gp_test',
    ext_modules = [
    Extension('gaussian_process',
        sources = ['gaussian_process.cc','../src/linalg.cc'],
        include_dirs = ['../include'],
        language = 'c++14',
        extra_compile_args = sysconfig.get_config_var('CFLAGS').split() +
            ['-std=c++14','-Wall','-O3'],
        extra_link_args = ['-lstdc++']
    )
])
