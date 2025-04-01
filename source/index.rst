.. pyCLINE documentation master file, created by
   sphinx-quickstart on Tue Mar  4 13:58:26 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
.. image:: _static/pycline_logo.png
   :alt: pyCLINE Logo
   :align: center
   :width: 100px

PyCLINE - Python Package for CLINE documentation
================================================

The ``pyCLINE`` package is the Python package based on the CLINE 
(Computational Learning and Identification of Nullclines).
It can be downloaded from PyPI with pip by using:

.. code-block:: python

    pip install pyCLINE

The package allows recreating all results shown in the manuscript Prokop, Billen, Frolov, Gelens (2025), but also allows to use the CLINE method on self generated data sets either synthetic or experimental. In order to generate data used in 
manuscript  a set of different models is being provided under ``pyCLINE.model``. 
Data from these models can be generated using ``pyCLINE.generate_data()``.
For setting up the data preparation and adjacent training a neural network, 
the submodule ``pyCLINE.recovery_methods`` is used. 
The submodule contains the module for data preparation 
``pyCLINE.recovery_methods.data_preparation`` and for neural network 
training ``pyCLINE.recovery_methods.nn_training``.

For a better understanding, ``pyCLINE`` also contains the module 
``pyCLINE.example`` which provides four examples also found in the original manusript with 
step-by-step instructions on how to set up a CLINE pipeline.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage 
   modules
   contributing