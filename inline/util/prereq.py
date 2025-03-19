import sys
import torch.utils.cpp_extension as cpp_extension
cpp_extension._nt_quote_args = lambda args: args  # Workaround for Windows quoting issue

# Install VS if needed
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


#!TODO
def install():
    """Install Visual Studio Build Tools if needed directly from setuptools"""
    ...
    
    
    
if sys.platform == "win32":
    install()  # This installs VS if needed

# Then try your load_inline code