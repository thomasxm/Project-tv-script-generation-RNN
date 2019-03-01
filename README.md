[image1]: ./script.jpg "Sample Output"
# Project-tv-script-generation-RNN
This project implemented _RNN_ and _embedding_ to generates tv scripts based on the famous "Seinfeld" script from 9 seasons
## Introduction
In this project, you'll generate your own Seinfeld TV scripts using RNNs. You'll be using a Seinfeld dataset of scripts from 9 seasons. The Neural Network you'll build will generate a new, "fake" TV script.
![Sample Output][image1]

## Topics:
In this project, we cover several areas:
* Machine learning
* Natural language processing
* RNN (LSTM, GRU)
* Word embedding
* Pytorch

 The project can be extended to include the word2vec embedding (Skip Grams model, negative sampling): 
 
* Use validation data to choose the best model
* Initialize your model weights, especially the weights of the embedded layer to encourage model convergence
* Use topk sampling to generate new words
* Generate your own Bach music using like DeepBach [here](https://arxiv.org/pdf/1612.01010.pdf).
* Predict seizures in intracranial EEG recordings on Kaggle [here](https://www.kaggle.com/c/seizure-prediction).

## Configure and Manage Your Environment with Anaconda

Per the Anaconda [docs](http://conda.pydata.org/docs):

> Conda is an open source package management system and environment management system 
for installing multiple versions of software packages and their dependencies and 
switching easily between them. It works on Linux, OS X and Windows, and was created 
for Python programs but can package and distribute any software.

## Overview
Using Anaconda consists of the following:

1. Install [`miniconda`](http://conda.pydata.org/miniconda.html) on your computer, by selecting the latest Python version for your operating system. If you already have `conda` or `miniconda` installed, you should be able to skip this step and move on to step 2.
2. Create and activate * a new `conda` [environment](http://conda.pydata.org/docs/using/envs.html).

\* Each time you wish to work on any exercises, activate your `conda` environment!

---

## 1. Installation

**Download** the latest version of `miniconda` that matches your system.

|        | Linux | Mac | Windows | 
|--------|-------|-----|---------|
| 64-bit | [64-bit (bash installer)][lin64] | [64-bit (bash installer)][mac64] | [64-bit (exe installer)][win64]
| 32-bit | [32-bit (bash installer)][lin32] |  | [32-bit (exe installer)][win32]

[win64]: https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe
[win32]: https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86.exe
[mac64]: https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
[lin64]: https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
[lin32]: https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86.sh

**Install** [miniconda](http://conda.pydata.org/miniconda.html) on your machine. Detailed instructions:

- **Linux:** http://conda.pydata.org/docs/install/quick.html#linux-miniconda-install
- **Mac:** http://conda.pydata.org/docs/install/quick.html#os-x-miniconda-install
- **Windows:** http://conda.pydata.org/docs/install/quick.html#windows-miniconda-install

## 2. Create and Activate the Environment

For Windows users, these following commands need to be executed from the **Anaconda prompt** as opposed to a Windows terminal window. For Mac, a normal terminal window will work. 

#### Git and version control
These instructions also assume you have `git` installed for working with Github from a terminal window, but if you do not, you can download that first with the command:
```
conda install git
```

If you'd like to learn more about version control and using `git` from the command line, take a look at our [free course: Version Control with Git](https://www.udacity.com/course/version-control-with-git--ud123).

**Now, we're ready to create our local environment!**

1. Clone the repository, and navigate to the downloaded folder. This may take a minute or two to clone due to the included image data.
```
git clone https://github.com/thomasxmeng/Project-tv-script-generation-RNN.git
cd Project-tv-script-generation-RNN
```

2. Create (and activate) a new environment, named `deep-learning` with Python 3.6. If prompted to proceed with the install `(Proceed [y]/n)` type y.

	- __Linux__ or __Mac__: 
	```
	conda create -n deep-learning python=3.6
	source activate deep-learning
	```
	- __Windows__: 
	```
	conda create --name deep-learning python=3.6
	activate deep-learning
	```
	
	At this point your command line should look something like: `(deep-learning) <User>:Project-tv-script-generation-RNN <user>$`. The `(deep-learning)` indicates that your environment has been activated, and you can proceed with further package installations.

3. Install PyTorch and torchvision; this should install the latest version of PyTorch.
	
	- __Linux__ or __Mac__: 
	```
	conda install pytorch torchvision -c pytorch 
	```
	- __Windows__: 
	```
	conda install pytorch -c pytorch
	pip install torchvision
	```

6. Install a few required pip packages, which are specified in the requirements text file (including OpenCV).
```
pip install -r requirements.txt
```

7. That's it!

Now most of the `deep-learning` libraries are available to you. Very occasionally, you will see a repository with an addition requirements file, which exists should you want to use TensorFlow and Keras, for example. In this case, you're encouraged to install another library to your existing environment, or create a new environment for a specific project. 

Noe, assuming your `deep-learning` environment is still activated, you can navigate to the main repo and start looking at the notebook:

```
cd
cd Project-tv-script-generation-RNN
jupyter notebook
```

To exit the environment when you have completed your work session, simply close the terminal window (From Udacity)


## Udacity work space
### Best Practices
The following project iterates over a large amount of data, and it's expected to take a number of hours to train, even on GPU. Follow the best practices outlined below to avoid common issues with GPU Workspaces.

* Keeping your connection alive during long processes
Workspaces automatically disconnect when the connection is inactive for about 30 minutes, which includes inactivity while deep learning models are training. You can use the workspace_utils.py [here](workspace_utils.py) module here to keep your connection alive during training. The module provides a context manager and an iterator wrapper—see example use below. 

**NOTE**: The script sometimes raises a connection error if the request is opened too frequently; just restart the Jupyter kernel & run the cells again to reset the error. 

**NOTE**: These scripts will keep your connection alive while the training process is running, but the workspace will still disconnect 30 minutes after the last notebook cell finishes. Modify the notebook cells to save your work at the end of the last cell or else you'll lose all progress when the workspace terminates. 

#### Example using context manager:
```sh
from workspace_utils import active_session

with active_session():
    # do long-running work here
```
#### Example using iterator wrapper:
```sh
from workspace_utils import keep_awake

for i in keep_awake(range(5)):
    # do iteration with lots of work here
```
* Manage your GPU time
It is important to avoid wasting GPU time in Workspace projects that have GPU acceleration enabled. The benefits of GPU acceleration are most useful when evaluating deep learning models—especially during training. In most cases, you can build and test your model (including data pre-processing, defining model architecture, etc.) in CPU mode, then activate GPU mode to accelerate training.

* Handling "Out of Memory" errors
This issue isn't specific to Workspaces, but rather it is an apparent issue between PyTorch & Jupyter, where Jupyter reports "out of memory" after a cell crashes. Jupyter holds references to active objects as long as the kernel is running—including objects created before an error is raised. This can cause Jupyter to persist large objects in memory long after they are no longer required. The only known solutions are:

  - To reduce the batch_size of your data
  - Reset the kernel and run the notebook cells again
