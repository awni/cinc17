### Install

Install dependencies for running on the deep cluster with Python 3 and GPU enabled Tensorflow

```
wget https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py --user
$HOME/.local/bin/pip3 install virtualenv --user

# *NB* if you are on AFS you may not have enough space in your home directory
# for the environment. I recommend putting it in scratch or somewhere where 
# you have a few GB of space.
$HOME/.local/bin/virtualenv ecg_env
source ecg_env/bin/activate # add to .bashrc.user


pip install -r path_to/requirements.txt

## Add below to .bashrc.user
# for cuda 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64

# for cuda nvcc
export PATH=$PATH:/usr/local/cuda-8.0/bin:
```

### Run

Run with
```
gpu=0
env CUDA_VISIBLE_DEVICES=$gpu python train.py
```

### Tensorboard

To view results run:
```
port=8888
log_dir=<directory_of_saved_models>
tensorboard --port $port --logdir $log_dir
```

### Jupyter Notebook

First install `jupyter` with
```
pip install jupyter
```

Then to launch the notebook

```
cd notebooks
env CUDA_VISIBLE_DEVICES=<gpu> jupyter notebook --port <port> --ip 0.0.0.0
```
replace `<gpu>` and `<port>` with desired values.


