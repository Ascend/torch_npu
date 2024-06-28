import glob
import shutil
import multiprocessing
import os
import re
import stat
import subprocess
import sys
import traceback
import platform
import setuptools
import time
from pathlib import Path
from typing import Union

import distutils.command.clean
from sysconfig import get_paths
from distutils.version import LooseVersion
from setuptools import setup, distutils, Extension, find_packages, setup

from codegen.utils import PathManager

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
PathManager.check_directory_path_readable(os.path.join(BASE_DIR, "version.txt"))

with open(os.path.join(BASE_DIR, "version.txt")) as version_f:
    VERSION = version_f.read().strip()
UNKNOWN = "Unknown"

cwd = os.path.dirname(os.path.abspath(__file__))


def get_submodule_folders():
    git_modules_path = os.path.join(cwd, ".gitmodules")
    with open(git_modules_path) as f:
        return [
            os.path.join(cwd, line.split("=", 1)[1].strip())
            for line in f
            if line.strip().startswith("path")
        ]


def check_submodules():
    def check_for_files(folder, files):
        if not any(os.path.exists(os.path.join(folder, f)) for f in files):
            print("Could not find any of {} in {}".format(", ".join(files), folder))
            print("Did you run 'git submodule update --init --recursive'?")
            sys.exit(1)

    def not_exists_or_empty(folder):
        return not os.path.exists(folder) or (
            os.path.isdir(folder) and len(os.listdir(folder)) == 0
        )

    folders = get_submodule_folders()
    # If none of the submodule folders exists, try to initialize them
    if all(not_exists_or_empty(folder) for folder in folders):
        try:
            print(" --- Trying to initialize submodules")
            start = time.time()
            subprocess.check_call(
                ["git", "submodule", "update", "--init", "--recursive"], cwd=cwd
            )
            end = time.time()
            print(f" --- Submodule initialization took {end - start:.2f} sec")
        except Exception:
            print(" --- Submodule initalization failed")
            print("Please run:\n\tgit submodule update --init --recursive")
            sys.exit(1)
    for folder in folders:
        check_for_files(
            folder,
            [
                "CMakeLists.txt",
                "Makefile",
                "setup.py",
                "LICENSE",
                "LICENSE.md",
                "LICENSE.txt",
            ],
        )


def get_sha(root: Union[str, Path]) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root)
            .decode("ascii")
            .strip()
        )
    except Exception:
        return UNKNOWN


def generate_version(path: Union[str, Path]):
    root = Path(__file__).parent
    path = root.joinpath(path)
    if path.exists():
        path.unlink()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    modes = stat.S_IWUSR | stat.S_IRUSR
    sha = get_sha(root)
    if os.getenv("BUILD_WITHOUT_SHA") is None:
        global VERSION
        VERSION += "+git" + sha[:7]
    with os.fdopen(os.open(path, flags, modes), "w") as f:
        f.write("__version__ = '{version}'\n".format(version=VERSION))
        f.write("git_version = {}\n".format(repr(sha)))


def which(thefile):
    path = os.environ.get("PATH", os.defpath).split(os.pathsep)
    for d in path:
        fname = os.path.join(d, thefile)
        fnames = [fname]
        if sys.platform == "win32":
            exts = os.environ.get("PATHEXT", "").split(os.pathsep)
            fnames += [fname + ext for ext in exts]
        for name in fnames:
            if os.access(name, os.F_OK | os.X_OK) and not os.path.isdir(name):
                return name
    return None


def get_cmake_command():
    def _get_version(cmd):
        for line in (
            subprocess.check_output([cmd, "--version"]).decode("utf-8").split("\n")
        ):
            if "version" in line:
                return LooseVersion(line.strip().split(" ")[2])
        raise RuntimeError("no version found")

    "Returns cmake command."
    cmake_command = "cmake"
    if platform.system() == "Windows":
        return cmake_command
    cmake3 = which("cmake3")
    cmake = which("cmake")
    if cmake3 is not None and _get_version(cmake3) >= LooseVersion("3.18.0"):
        cmake_command = "cmake3"
        return cmake_command
    elif cmake is not None and _get_version(cmake) >= LooseVersion("3.18.0"):
        return cmake_command
    else:
        raise RuntimeError("no cmake or cmake3 with version >= 3.18.0 found")


def get_build_type():
    build_type = "Release"
    if os.getenv("DEBUG", default="0").upper() in ["ON", "1", "YES", "TRUE", "Y"]:
        build_type = "Debug"

    if os.getenv("REL_WITH_DEB_INFO", default="0").upper() in [
        "ON",
        "1",
        "YES",
        "TRUE",
        "Y",
    ]:
        build_type = "RelWithDebInfo"

    return build_type


def get_pytorch_dir():
    try:
        import torch

        return os.path.dirname(os.path.realpath(torch.__file__))
    except Exception:
        _, _, exc_traceback = sys.exc_info()
        frame_summary = traceback.extract_tb(exc_traceback)[-1]
        return os.path.dirname(frame_summary.filename)


def generate_bindings_code(base_dir):
    python_execute = sys.executable
    generate_code_cmd = [
        "bash",
        os.path.join(base_dir, "generate_code.sh"),
        python_execute,
        VERSION,
    ]
    if subprocess.call(generate_code_cmd) != 0:  # Compliant
        print(
            "Failed to generate ATEN bindings: {}".format(generate_code_cmd),
            file=sys.stderr,
        )
        sys.exit(1)


def CppExtension(name, sources, *args, **kwargs):
    r"""
    Creates a :class:`setuptools.Extension` for C++.
    """
    pytorch_dir = get_pytorch_dir()
    temp_include_dirs = kwargs.get("include_dirs", [])
    temp_include_dirs.append(os.path.join(pytorch_dir, "include"))
    temp_include_dirs.append(
        os.path.join(pytorch_dir, "include/torch/csrc/api/include")
    )
    kwargs["include_dirs"] = temp_include_dirs

    temp_library_dirs = kwargs.get("library_dirs", [])
    temp_library_dirs.append(os.path.join(pytorch_dir, "lib"))
    kwargs["library_dirs"] = temp_library_dirs

    libraries = kwargs.get("libraries", [])
    libraries.append("c10")
    # libraries.append("torch")
    libraries.append("torch_cpu")
    libraries.append("torch_python")
    # libraries.append("hccl")
    kwargs["libraries"] = libraries
    kwargs["language"] = "c++"
    return Extension(name, sources, *args, **kwargs)


class Clean(distutils.command.clean.clean):

    def run(self):
        f_ignore = open(".gitignore", "r")
        ignores = f_ignore.read()
        pat = re.compile(r"^#( BEGIN NOT-CLEAN-FILES )?")
        for wildcard in filter(None, ignores.split("\n")):
            match = pat.match(wildcard)
            if match:
                if match.group(1):
                    # Marker is found and stop reading .gitignore.
                    break
                # Ignore lines which begin with '#'.
            else:
                for filename in glob.glob(wildcard):
                    if os.path.islink(filename):
                        raise RuntimeError(f"Failed to remove path: {filename}")
                    if os.path.exists(filename):
                        try:
                            shutil.rmtree(filename, ignore_errors=True)
                        except Exception as err:
                            raise RuntimeError(
                                f"Failed to remove path: {filename}"
                            ) from err
        f_ignore.close()

        # It's an old-style class in Python 2.7...
        distutils.command.clean.clean.run(self)

        remove_files = [
            "aten/RegisterCPU.cpp",
            "aten/RegisterNPU.cpp",
            "aten/RegisterAutogradNPU.cpp",
            "aten/NPUNativeFunctions.h",
            "aten/CustomRegisterSchema.cpp",
            "aten/ForeachRegister.cpp",
            "torch_npu/utils/custom_ops.py",
            "torch_npu/version.py",
        ]
        for remove_file in remove_files:
            file_path = os.path.join(BASE_DIR, remove_file)
            if os.path.exists(file_path):
                os.remove(file_path)


def get_src_py_and_dst():
    ret = []

    # ret = glob.glob(
    #     os.path.join(BASE_DIR, "torch_npu", '**/*.yaml'),
    #     recursive=True) + glob.glob(
    #     os.path.join(BASE_DIR, "torch_npu", 'acl.json'),
    #     recursive=True) + glob.glob(
    #     os.path.join(BASE_DIR, "torch_npu", 'contrib/apis_config.json'),
    #     recursive=True)

    header_files = [
        "third_party/acl/inc/*/*.h",
        "third_party/acl/inc/*/*/*.h",
        "third_party/op-plugin/op_plugin/include/*.h",
    ]
    glob_header_files = []
    for regex_pattern in header_files:
        glob_header_files += glob.glob(
            os.path.join(BASE_DIR, regex_pattern), recursive=True
        )

    for src in glob_header_files:
        dst = os.path.join(
            os.path.join(BASE_DIR, "torch_npu/include/third_party"),
            os.path.relpath(src, os.path.join(BASE_DIR, "third_party")),
        )
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        ret.append((src, dst))

    header_files = [
        "torch_npu/csrc/*.h",
        "torch_npu/csrc/*/*.h",
        "torch_npu/csrc/*/*/*.h",
        "torch_npu/csrc/*/*/*/*.h",
        "torch_npu/csrc/*/*/*/*/*.h",
    ]
    glob_header_files = []
    for regex_pattern in header_files:
        glob_header_files += glob.glob(
            os.path.join(BASE_DIR, regex_pattern), recursive=True
        )

    for src in glob_header_files:
        dst = os.path.join(
            os.path.join(BASE_DIR, "torch_npu/include/torch_npu"),
            os.path.relpath(src, os.path.join(BASE_DIR, "torch_npu")),
        )
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        ret.append((src, dst))

    for src, dst in ret:
        shutil.copyfile(src, dst)


def build_deps():
    check_submodules()
    generate_bindings_code(BASE_DIR)

    cmake = get_cmake_command()

    if cmake is None:
        raise RuntimeError(
            "CMake must be installed to build the following extensions: "
            + ", ".join(e.name for e in self.extensions)
        )

    build_dir = os.path.join(BASE_DIR, "build")
    build_type_dir = os.path.join(build_dir)
    os.makedirs(build_type_dir, exist_ok=True)

    output_lib_path = os.path.join(BASE_DIR, "torch_npu")
    os.makedirs(output_lib_path, exist_ok=True)

    cmake_args = [
        "-DCMAKE_BUILD_TYPE=" + get_build_type(),
        "-DCMAKE_INSTALL_PREFIX=" + os.path.realpath(output_lib_path),
        "-DPYTHON_INCLUDE_DIR=" + get_paths().get("include"),
        "-DTORCH_VERSION=" + VERSION,
        "-DPYTORCH_INSTALL_DIR=" + get_pytorch_dir(),
    ]

    subprocess.check_call(
        [cmake, BASE_DIR] + cmake_args, cwd=build_type_dir, env=os.environ
    )

    build_args = [
        "--build",
        ".",
        "--target",
        "install",
        "--",
    ]

    build_args += ["-j", str(multiprocessing.cpu_count())]

    command = [cmake] + build_args
    subprocess.check_call(command, cwd=build_type_dir, env=os.environ)

    get_src_py_and_dst()


def configure_extension_build():
    include_directories = [
        BASE_DIR,
        os.path.join(BASE_DIR, "npu/acl/include"),
    ]

    extra_link_args = []

    DEBUG = os.getenv("DEBUG", default="").upper() in ["ON", "1", "YES", "TRUE", "Y"]

    extra_compile_args = [
        "-std=c++17",
        "-Wno-sign-compare",
        "-Wno-deprecated-declarations",
        "-Wno-return-type",
    ]

    if re.match(r"clang", os.getenv("CC", "")):
        extra_compile_args += [
            "-Wno-macro-redefined",
            "-Wno-return-std-move",
        ]

    if DEBUG:
        extra_compile_args += ["-O0", "-g"]
        extra_link_args += ["-O0", "-g", "-Wl,-z,now"]
    else:
        extra_compile_args += ["-DNDEBUG"]
        extra_link_args += ["-Wl,-z,now"]

    excludes = ["codegen", "codegen.*"]
    packages = find_packages(exclude=excludes)

    extension = []
    C = CppExtension(
        "torch_npu._C",
        sources=["torch_npu/csrc/stub.c"],
        libraries=["torch_npu"],
        include_dirs=include_directories,
        extra_compile_args=extra_compile_args
        + ["-fstack-protector-all"]
        + ['-D__FILENAME__="stub.c"'],
        library_dirs=["lib", os.path.join(BASE_DIR, "torch_npu/lib")],
        extra_link_args=extra_link_args + ["-Wl,-rpath,$ORIGIN/lib"],
        define_macros=[("_GLIBCXX_USE_CXX11_ABI", "0"), ("GLIBCXX_USE_CXX11_ABI", "0")],
    )
    extension.append(C)

    cmdclass = {
        "clean": Clean,
    }

    return extension, cmdclass, packages


VERBOSE_SCRIPT = True
RUN_BUILD_DEPS = True

filtered_args = []
for i, arg in enumerate(sys.argv):
    if arg == "--":
        filtered_args += sys.argv[i:]
        break
    if arg == "-q" or arg == "--quiet":
        VERBOSE_SCRIPT = False
    if arg in ["clean", "egg_info", "sdist"]:
        RUN_BUILD_DEPS = False
    filtered_args.append(arg)
sys.argv = filtered_args

if VERBOSE_SCRIPT:

    def report(*args):
        print(*args)

else:

    def report(*args):
        pass

    setuptools.distutils.log.warn = report


def main():
    install_requires = ["pyyaml"]

    if sys.version_info >= (3, 12, 0):
        install_requires.append("setuptools")

    dist = setuptools.dist.Distribution()
    dist.script_name = os.path.basename(sys.argv[0])
    dist.script_args = sys.argv[1:]
    try:
        dist.parse_command_line()
    except setuptools.distutils.errors.DistutilsArgError as e:
        report(e)
        sys.exit(1)

    generate_version("torch_npu/version.py")

    if RUN_BUILD_DEPS:
        build_deps()

    (
        extensions,
        cmdclass,
        packages,
    ) = configure_extension_build()

    readme = os.path.join(BASE_DIR, "README.md")
    if not os.path.exists(readme):
        raise FileNotFoundError("Unable to find 'README.md'")
    with open(readme, encoding="utf-8") as fdesc:
        long_description = fdesc.read()

    setup(
        name=os.environ.get("TORCH_NPU_PACKAGE_NAME", "torch_npu"),
        version=VERSION,
        description="NPU bridge for PyTorch",
        long_description=long_description,
        long_description_content_type="text/markdown",
        ext_modules=extensions,
        cmdclass=cmdclass,
        packages=packages,
        package_data={
            "torch_npu": [
                "*.so",
                "lib/*.so*",
            ],
        },
        install_requires=install_requires,
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Mathematics",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Software Development",
            "Topic :: Software Development :: Libraries",
            "Topic :: Software Development :: Libraries :: Python Modules",
            "Programming Language :: C++",
            "Programming Language :: Python :: 3",
        ],
        license="BSD License",
        keywords="pytorch, machine learning",
    )


if __name__ == "__main__":
    main()
