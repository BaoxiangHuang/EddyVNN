# EddyVNN
This code is come from paper "Vertical Structure based Classification of Oceanic Eddy using 3D Convolutional Neural Network"

## Dateset
In the process of experiments, we adopted Argo data set and XBT data set from 2002 to 2018, which are provided by the dataset (Data ID:XDA19090202 002).
However, the size of the pre-processed data is too large so that we could not upload the data. Therefore, please download the data for yourself.

## Dependency Libraries
The code is built on Pytorch and tested on Ubuntu 18.04 LTS.
The dependency Libraries are as follow:

> pytorch = 1.4.0  
> scikit-learn = 0.23.1  
> numpy = 1.19.4  

## Code Structure
The whole code included three parts: "Model.py", "net_main.py" and "process.py". The "Model.py" includes the EddyVNN model structure. The "process.py" contains process of reading data and normalization, and the interpolation method.
