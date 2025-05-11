import os
import platform
import sys

import numpy as np
import pybind11
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


# 自动收集 src 目录下的所有 C++ 源文件
def find_sources(src_dir):
    sources = []
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith(".cpp"):
                sources.append(os.path.join(root, file))
    return sources


# 核心配置
__version__ = '0.1.0'
MODULE_NAME = 'deepsearch'
BINDINGS_DIR = "python_bindings"

# 获取 src 目录的绝对路径（根据实际位置调整）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINDINGS_DIR = os.path.join(project_root, BINDINGS_DIR)
cpp_src_dir = os.path.join(project_root, "src")
cpp_sources = find_sources(cpp_src_dir)

# 包含目录配置
include_dirs = [
    pybind11.get_include(),
    np.get_include(),
    os.path.join(project_root, "src"),  # 确保头文件路径正确
]

# 绑定文件 + 所有 C++ 源文件
source_files = [
    os.path.join(BINDINGS_DIR, 'bindings.cpp'),
]
source_files.extend(cpp_sources)

# 编译模块配置
ext_modules = [
    Extension(
        MODULE_NAME,
        sources=source_files,
        include_dirs=include_dirs,
        language='c++',
        extra_compile_args=['-std=c++17', '-O3'],  # 添加优化和 C++17 支持
    ),
]

# 系统检测
IS_LINUX = platform.system() == 'Linux'
IS_MACOS = platform.system() == 'Darwin'
IS_WINDOWS = platform.system() == 'Windows'


class BuildExt(build_ext):
    """自动检测并配置 OpenMP 支持，处理 macOS 的 libomp 路径"""
    user_options = build_ext.user_options + [
        ('disable-openmp', None, "Disable OpenMP support"),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.disable_openmp = False
        self.openmp_include_dir = None
        self.openmp_library_dir = None

    def finalize_options(self):
        super().finalize_options()
        # 从环境变量中读取自定义 OpenMP 路径
        self.openmp_include_dir = os.environ.get('OPENMP_INCLUDE_DIR')
        self.openmp_library_dir = os.environ.get('OPENMP_LIBRARY_DIR')

    def build_extensions(self):
        # 设置 C++ 标准
        cpp_std = '-std=c++17'
        if IS_WINDOWS:
            cpp_std = '/std:c++17' if sys.version_info >= (3, 9) else '/std:c++latest'

        # 公共编译选项
        for ext in self.extensions:
            ext.extra_compile_args.append(cpp_std)
            if not IS_WINDOWS:
                ext.extra_compile_args.append(f'-DVERSION_INFO="{self.distribution.get_version()}"')
                ext.extra_compile_args.append('-fvisibility=hidden')
            else:
                ext.extra_compile_args.append(f'/DVERSION_INFO=\\"{self.distribution.get_version()}\\"')

        # OpenMP 支持检测
        openmp_supported = False
        if not self.disable_openmp:
            openmp_supported = self._check_openmp_support()
            print(f"OpenMP support: {'Enabled' if openmp_supported else 'Disabled'}")

        # 配置 OpenMP 选项
        if openmp_supported:
            self._add_openmp_flags()

        super().build_extensions()

    def _check_openmp_support(self):
        """全面检测 OpenMP 支持（编译 + 链接）"""
        test_code = r"""
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

            # 生成测试文件
            test_file = os.path.join(tmp_dir, 'openmp_test.cpp')
            with open(test_file, 'w') as f:
                f.write(test_code)

            # 获取编译和链接参数（包含路径）
            compile_args = self._get_openmp_compile_args()
            link_args = self._get_openmp_link_args()

            # 添加自定义包含路径和库路径
            if self.openmp_include_dir:
                compile_args.extend(['-I', self.openmp_include_dir])
            if self.openmp_library_dir:
                link_args.extend(['-L', self.openmp_library_dir])

            # macOS 默认 Homebrew 路径
            if IS_MACOS and not self.openmp_include_dir:
                brew_omp_include = '/opt/homebrew/opt/libomp/include'
                if os.path.exists(brew_omp_include):
                    compile_args.extend(['-I', brew_omp_include])
            if IS_MACOS and not self.openmp_library_dir:
                brew_omp_lib = '/opt/homebrew/opt/libomp/lib'
                if os.path.exists(brew_omp_lib):
                    link_args.extend(['-L', brew_omp_lib])

            # 编译测试程序
            objects = self.compiler.compile(
                [test_file],
                output_dir=tmp_dir,
                extra_postargs=compile_args
            )

            # 链接测试程序
            self.compiler.link_executable(
                objects,
                os.path.join(tmp_dir, 'openmp_test'),
                extra_postargs=link_args
            )

            return True
        except (CompileError, LinkError) as e:
            print(f"OpenMP 检测失败: {str(e)}")
            return False
        except Exception as e:
            print(f"OpenMP 检测异常: {str(e)}")
            return False
        finally:
            # 清理临时文件
            pass  # 可根据需要添加清理逻辑

    def _add_openmp_flags(self):
        """添加平台相关的 OpenMP 编译和链接选项"""
        for ext in self.extensions:
            # 编译选项
            if IS_MACOS:
                ext.extra_compile_args.extend(['-Xpreprocessor', '-fopenmp'])
                # 添加自定义包含路径
                if self.openmp_include_dir:
                    ext.include_dirs.append(self.openmp_include_dir)
                else:
                    # 默认 Homebrew 路径
                    brew_omp_include = '/opt/homebrew/opt/libomp/include'
                    if os.path.exists(brew_omp_include):
                        ext.include_dirs.append(brew_omp_include)
            elif IS_LINUX:
                ext.extra_compile_args.append('-fopenmp')
            elif IS_WINDOWS:
                ext.extra_compile_args.append('/openmp')

            # 链接选项
            if IS_MACOS:
                ext.extra_link_args.append('-lomp')
                # 添加自定义库路径
                if self.openmp_library_dir:
                    ext.library_dirs.append(self.openmp_library_dir)
                else:
                    # 默认 Homebrew 路径
                    brew_omp_lib = '/opt/homebrew/opt/libomp/lib'
                    if os.path.exists(brew_omp_lib):
                        ext.library_dirs.append(brew_omp_lib)
            elif IS_LINUX:
                ext.extra_link_args.append('-fopenmp')

            # 定义宏
            # ext.define_macros.append(('_OPENMP', None))

    def _get_openmp_compile_args(self):
        """获取平台特定的 OpenMP 编译参数"""
        args = []
        if IS_MACOS:
            args = ['-Xpreprocessor', '-fopenmp']
        elif IS_LINUX:
            args = ['-fopenmp']
        elif IS_WINDOWS:
            args = ['/openmp']
        return args

    def _get_openmp_link_args(self):
        """获取平台特定的 OpenMP 链接参数"""
        args = []
        if IS_MACOS:
            args = ['-lomp']
        elif IS_LINUX:
            args = ['-fopenmp']
        return args


setup(
    name='deepsearch',
    version=__version__,
    description='Deep Approximate Nearest Neighbor Search',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    author='Koschei',
    author_email='nitianzero@gmail.com',
    url='https://github.com/kosthi/deepsearch',
    ext_modules=ext_modules,
    install_requires=[
        'numpy>=1.18',
        'pybind11>=2.6'
    ],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'deepsearch-cli=deepsearch.cli:main',
        ],
    },
    include_package_data=True,
)
