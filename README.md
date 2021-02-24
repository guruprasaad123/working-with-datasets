





# Using GPU

### Pytorch

After installing the nvidia driver from [here](https://developer.nvidia.com/cuda/wsl/download) , we will be able to access gpu via cuda ( through WSL2 )

```bash
Python 3.7.9 (default, Aug 31 2020, 12:42:55)
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> torch.cuda.is_available()
True
>>> torch.cuda.device_count()
1
>>> torch.cuda.device(0)
<torch.cuda.device object at 0x7f4d2b18b1d0>
>>> torch.cuda.get_device_name()
'GeForce GTX 1050'
>>>

```

### Reference

https://pytorch.org/get-started/locally/#windows-verification

https://towardsdatascience.com/cuda-on-wsl2-for-deep-learning-first-impressions-and-benchmarks-3c0178754035

https://developer.nvidia.com/cuda/wsl/download

https://developer.nvidia.com/?destination=node/874687&autologout_timeout=1

https://docs.nvidia.com/cuda/wsl-user-guide/index.html

https://ngc.nvidia.com/catalog/collections

https://docs.microsoft.com/en-us/windows/win32/direct3d12/gpu-tensorflow-wsl

https://github.com/tensorflow/docs/tree/master/site/en/r1/tutorials

https://docs.microsoft.com/en-us/windows/wsl/install-win10#step-4---download-the-linux-kernel-update-package