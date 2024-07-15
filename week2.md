# Week 2: Learning Pytorch
##  Check if PyTorch can utilize the GPU

```python
import torch

if torch.cuda.is_available():
    print("CUDA is available.")
else:
    print("CUDA is not available.")
```
The output shows that pytorch cannot call CUDA

###  Search the cause

#### Check whether the environment variables are configured correctly
`sudo gedit ~/.hashrc`
- The environment variables are configured correctly

####  Reinstall PyTorch
- Create a virtual environment for Python 3.9
- Reinstall pytorch in the new environment
`conda install pytorch torchvision torchaudio cudatoolkit=12.1 -c pytorch -c nvidia`
- Still unable to call CUDA

#### Lower the CUDA and cdDNN version
- Falied

####  Check the environment in conda
`conda activate py312`
`codna list`
- Found that there is no pytorch GPU version in the conda environment list.
-  Go to conda Tsinghua source to find the corresponding pytorch, torchvision, torchaudio gpu versions.
 - Install these three packs
`conda install --offline pytorch-2.1.0-py3.12_cuda12.1_cudnn8_0.tar.bz2`
`conda install --offline torchaudio-2.1.0-py312_cu121.tar.bz2`
`conda install --offline torchvision-0.16.0-py312_cu121.tar.bz2`
- Check again whether CUDA can be called
- Finally success
