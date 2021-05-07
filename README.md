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
The whole code included three parts: "Model.py", "net_main.py" and "process.py". The "Model.py" includes the EddyVNN model structure. The "process.py" contains a process of reading data and normalization and the interpolation method. The "net_main.py" is the main operating file, achieving training and testing of models. In the processing of the training, we used 60% of the data as the training set and 40% of the data is used as the testing set.

## Running
run the following scripts to train EddyVNN model and test EddyVNN model. 
```
python net.main.py
```
you can run the following scripts to change the batch size, learning rate and other parameters.
```
python net.main.py --batch 512  --start_year 2002 --end_year 2010 --model_layers 101
```






