import os
import platform
from distutils.errors import CompileError, LinkError

import numpy as np
import pybind11
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# 项目信息
__version__ = '0.1.0'
MODULE_NAME = 'deepsearch'
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
BINDINGS_DIR = os.path.join(PROJECT_ROOT, "python_bindings")

# 平台信息
IS_WINDOWS = platform.system() == "Windows"
IS_MACOS = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"


def find_cpp_sources(directory):
    return [
        os.path.join(root, file)
        for root, _, files in os.walk(directory)
        for file in files if file.endswith(".cpp")
    ]


def get_openmp_flags():
    if IS_WINDOWS:
        return ["/openmp"], []
    elif IS_MACOS:
        return ["-Xpreprocessor", "-fopenmp"], ["-lomp"]
    elif IS_LINUX:
        return ["-fopenmp"], ["-fopenmp"]
    return [], []


class BuildExt(build_ext):
    user_options = build_ext.user_options + [('disable-openmp', None, "Disable OpenMP support")]

    def initialize_options(self):
        super().initialize_options()
        self.disable_openmp = False

    def finalize_options(self):
        super().finalize_options()
        self.openmp_include_dir = os.environ.get('OPENMP_INCLUDE_DIR')
        self.openmp_library_dir = os.environ.get('OPENMP_LIBRARY_DIR')
        if IS_MACOS:
            if not self.openmp_include_dir:
                self.openmp_include_dir = '/opt/homebrew/opt/libomp/include'
            if not self.openmp_library_dir:
                self.openmp_library_dir = '/opt/homebrew/opt/libomp/lib'

    def build_extensions(self):
        cpp_flag = '/std:c++17' if IS_WINDOWS else '-std=c++17'

        for ext in self.extensions:
            ext.extra_compile_args = [cpp_flag]
            ext.include_dirs.extend([
                pybind11.get_include(),
                np.get_include(),
                SRC_DIR
            ])
            if not IS_WINDOWS:
                ext.extra_compile_args += [
                    f'-DVERSION_INFO="{self.distribution.get_version()}"',
                    '-fvisibility=hidden'
                ]
            else:
                ext.extra_compile_args += [f'/DVERSION_INFO=\\"{self.distribution.get_version()}\\"']

            if not self.disable_openmp and self._check_openmp():
                compile_flags, link_flags = get_openmp_flags()
                ext.extra_compile_args += compile_flags
                ext.extra_link_args += link_flags

                # 包含和库路径设置
                if self.openmp_include_dir:
                    ext.include_dirs.append(self.openmp_include_dir)
                elif IS_MACOS:
                    ext.include_dirs.append('/opt/homebrew/opt/libomp/include')

                if self.openmp_library_dir:
                    ext.library_dirs.append(self.openmp_library_dir)
                elif IS_MACOS:
                    ext.library_dirs.append('/opt/homebrew/opt/libomp/lib')

        super().build_extensions()

    def _check_openmp(self):
        """尝试编译测试程序以检测 OpenMP 支持"""
        test_code = """
        #include <omp.h>
        int main() {
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
            }
            return 0;
        }
        """
        try:
            tmp_dir = self.build_temp
            os.makedirs(tmp_dir, exist_ok=True)
            test_file = os.path.join(tmp_dir, "test_openmp.cpp")
            with open(test_file, "w") as f:
                f.write(test_code)

            compile_args, link_args = get_openmp_flags()
            if self.openmp_include_dir:
                compile_args += ["-I", self.openmp_include_dir]
            if self.openmp_library_dir:
                link_args += ["-L", self.openmp_library_dir]

            objs = self.compiler.compile([test_file], output_dir=tmp_dir, extra_postargs=compile_args)
            self.compiler.link_executable(objs, os.path.join(tmp_dir, "test_openmp_exec"), extra_postargs=link_args)
            return True
        except (CompileError, LinkError, Exception) as e:
            print(f"OpenMP support test failed: {e}")
            return False


# 构建 Extension 模块
source_files = [os.path.join(BINDINGS_DIR, "bindings.cpp")] + find_cpp_sources(SRC_DIR)

ext_modules = [
    Extension(
        MODULE_NAME,
        sources=source_files,
        include_dirs=[],  # 会在 build_ext 中添加
        language="c++"
    )
]

# 安装配置
setup(
    name=MODULE_NAME,
    version=__version__,
    description='Deep Approximate Nearest Neighbor Search',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    author='Koschei',
    author_email='nitianzero@gmail.com',
    url='https://github.com/kosthi/deepsearch',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    install_requires=[
        'numpy>=1.18',
        'pybind11>=2.6'
    ],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache-2.0 License',
        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.6',
    zip_safe=False,
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'deepsearch-cli=deepsearch.cli:main',
        ],
    },
)
