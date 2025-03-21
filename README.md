# PyCLINE - python package for CLINE

The `pyCLINE` package is the python package based on the CLINE (**C**omputational **L**earning and **I**dentification of **N**ullclin**E**s).
It can be downloaded from PyPI with pip by using
    
    pip install pyCLINE

The package allows to recreate all data, models and results shown in [Prokop, Billen, Frolov and Gelens (2025)](https://arxiv.org/abs/2503.16240), and to apply CLINE to other data sets. 
In order to generate data used in [Prokop, Billen, Frolov and Gelens (2025)](https://arxiv.org/abs/2503.16240), a set of different models is being provided under `pyCLINE.model`. 
Data from these models can be generated using `pyCLINE.generate_data()`.
For setting up the data prepartion and adjacent training a neural network, the submodule `pyCLINE.recovery_methods` is used. 
The submodule contains the module for data_preparation `pyCLINE.recovery_methods.data_preparation` and for neural network training `pyCLINE.recovery_methods.nn_training`. 

For a better understanding, `pyCLINE` also contains the module `pyCLINE.example` which provides four examples also found in XXX with step by step instructions on how to setup a CLINE pipeline. 

The structure of `pyCLINE` is shown here: 

![PyCLINE structure](pycline_structure.png)