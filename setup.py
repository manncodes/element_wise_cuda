from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="element_wise_cuda",
    version="0.1",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="element_wise_cuda.element_wise_cuda",
            sources=[
                "element_wise_cuda/csrc/element_wise.cpp",
                "element_wise_cuda/csrc/element_wise_kernel.cu",
            ],
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
    install_requires=[
        "torch>=1.7.0",
    ],
)
