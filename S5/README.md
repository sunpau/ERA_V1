#  S5 Assignment

The S5 assignment trains a neural network on the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset. This contains the following files:
*  model.py
*  utils.py
*  S5.ipynb 

#  Installation

It is recommended to use google colab for this assignment. Download the files and place them in your google drive folder and set the directory to the folder.

# File Details
The function details in the files are:
*  model.py  
   *  The neural network is defined here. To add or modify layers modify the class Net().
   *  The functions train() and test() trains the network on the dataset and returns train/test accuracy and loss
    
* utils.py  
  *  The transforms to be applied to the data is defined in data_transforms() function.
  *  The functions to plot the data and the train/test accuracy are defined in plot_data(), plot_loss_accuracy()

*  S5.ipynb
   *  This file is the main file where the above .py files are imported as helpers.
