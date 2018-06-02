import glob
from os import path as osp
import torch
from torch.utils.ffi import create_extension

abs_path = osp.dirname(osp.realpath(__file__))

sources = ['src/andor.c']
headers = ['include/andor.h']
defines = []
with_cuda = False
extra_objects = [osp.join(abs_path, 'build/andor_cuda_kernel.so')]
extra_objects += glob.glob('/usr/local/cuda/lib64/*.a')

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/andor_cuda.c']
    headers += ['include/andor_cuda.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True

ffi = create_extension(
    '_ext.andor',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    # extra_compile_args=["-std=c99"],
    extra_objects=extra_objects,
    include_dirs=[osp.join(abs_path, 'include')]
)

if __name__ == '__main__':
    ffi.build()
