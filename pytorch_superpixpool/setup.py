from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

# setup(name='spxp',
#       ext_modules=[CppExtension('suppixpool_clean', ['suppixpool_cuda.cpp'])],
#       cmdclass={'build_ext': BuildExtension})

setup(
    name='suppixpool_CUDA',
    ext_modules=[
        CUDAExtension('suppixpool_CUDA', [
            'suppixpool_cuda.cpp',
            'suppixpool_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })