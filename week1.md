# Week 1: Setting Up Deep Learning Environment with PyTorch

## Step 1: Install GPU Driver

### Install NVIDIA GPU driver version 555 for GTX 4060 Ti on Ubuntu 22.04.

- Verify driver version on:
`https://www.nvidia.cn/Download/index.aspx?lang=cn`

- Disable Nouveau (Nouveau is the generic driver)
- Install GPU driver
`sudo chmod 777 NVIDIA-Linux-x86_64-555.26.run`
` sudo ./NVIDIA-Linux-x86_64-555.26.run （–no-opengl-files`

 

### Ensure GPU driver installation is successful.
- Open terminal
`nvidia-smi`
- The GPU driver is installed successfully.

## Step 2: Install CUDA Toolkit
### Install CUDA
- Verify on `https://developer.nvidia.com/cuda-toolkit-archive`

- Choose CUDA version `cuda_12.5.1_555.42.06_linux`
- Install `sudo sh cuda_11.6.2_510.47.03_linux.run`

### Configuring environment variables
- Open `~/.bashrc`   
`sudo gedit ~/.bahsrc` 
-Add at the end of the file
`export  PATH=$PATH:/usr/local/cuda/bin`
`export  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64`
- Reload `~/.bashrc`
`source ~/.bashrc`

### Ensure CUDA installation is successful.
- `nvcc -V`
The  CUDA is installed successfully
## Step 3: Install cuDNN

- Verify on `https://developer.nvidia.com/rdp/cudnn-archive#a-collapse805-111`
- Install `cudnn-linux-x86_64-9.1.1.17`.
- .
- .
- .
- Ensure
`cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2`

## Step 4: Install Anaconda3

- Create a Python 3.12 virtual environment using Anaconda:


## Step 5: Install PyTorch

- Install PyTorch and related libraries within the virtual environment using Conda:
`conda install pytorch torchvision torchaudio cudatoolkit=12.1 -c pytorch -c nvidia`
