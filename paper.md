---
title: 'pyCLINE: Python implementation of CLINE method for nullcline structure discovery'
tags:
  - Python
  - nonlinear dynamics
  - dynamics
  - machine learning
  - data-driven methods
  - model identification
authors:
  - name: Bartosz Prokop
    orcid: 0000-0001-9723-0176
    corresponding: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Jimmy Billen
    affiliation: 1
  - name: Nikit Frolov # (This is how to denote the corresponding author)
    affiliation: 1
    orcid: 0000-0002-2788-1907
  - name: Lendert Gelens
    corresponding: true
    affiliation: 1
    orcid: 0000-0001-7290-9561
affiliations:
 - name: Laboratory of Dynamics in Biological Systems, Department of Cellular and Mollecular Medicine, KU Leuven
   index: 1
date: 14 February 2025
bibliography: paper.bib
---

# Summary

Many important processes in physics, biology, chemistry and engineering are dynamic, just to mention some examples such as the orbiting of planets around the sun, the preiodic oscillations of the cell cycle, the Belousov-Zhabotinsky reaction or movements of fluids. 
Therefore, understanding and describing these processes is high relevant, and has been done in the form of differential equations for centuries. 
Usually, deriving such equations requires a mix of analysis of gathered data, established modelling principles and scientific intuition. 
However, this approach can be difficult or even impossible due to either large data sets or false assumpotions of principles due to bias `[@Murari:2020,@Prokop:2024]`. 
To tackle this challenge, many data-driven or also called machine learning approaches have been suggested in recent years, that allow to derive symbolic or form-free models directly from measured data, e.g. Sparse Identification of nonlinear dynamics (SINDy) 
`[@Brunton:2016]`, Nonlinear Autoregressive Moving Average Model with Exogenous Inputs (NARMAX),Symbolic Regression (SR)`[@Cranmer:2019]`, Reservoir Computing for dynamical systmes `[@Haluszczynski:2019]`, Physics-Informed neural networks (PINN)`[@Karnadiakis:2021]`, biology-informed neural networks (BINNs) `[@Lagergren:2020]`, Universal Differential Equations `[@Rackauckas:2020]` and many more. 

Most of the existing methods focus mostly on forecasting in order to determine an adequate model that is able to adequately describe the temporal evolution of a system. 
In contrast to that, we have developed a new method called CLINE (**C**omputational **L**earning and **I**nference of **N**ullcline **E**quations) `[@Prokop:2025]`, that instead focuses on identifying the underlying static information in the phase space such as the nullcline structure. 
Knowledge of the nullcline structure has many advantages `[@ProkopB:2024]`: 
- Nullclines fully describe the dynamical systems behavior and therefore provide more information then just the time series
- When the structure of nullclines is known, the symbolic equation can be derived using symbolic model identification methods, but applied to a problem of signficantly lower complexity then time series data
- Identifying the structure without a predefine set of candidate terms (e.g. in form of a library) but model-free allows to avoid implementation of false bias in the model formulation

# Statement of need

`pyCLINE` is a Python package that allows to easily set up and use the CLINE method as explained and shown in `@Prokop:2025`. It is based on the Python torch implementation `pyTorch` and allows to quickly implement the identification of nullcline structures from simulated or measured time series data. 
The implementation of `pyCLINE` allows to generate examplary data sets from scratch, correctly prepare data for training and set up the feed forward neural network for training. 

`pyCLINE` was designed to be used by researchers experienced with the use of machine learning or laymen that are interested to apply the method to either different models or measured data. 
This allows for simple and fast implementation in many fields that are interested in discovering nullcline structures from measured data, that can help develop novel or provide proof for existing models of dynamic (oscillatory) systems.

# Usage

The `pyCLINE` package includes three main modules (see \autoref{fig:method}): 
 - `pyCLINE.generate_data()`: This module generates data whifh has been used in `@Prokop:2025` with additionally many more models that can be found under `pyCLINE.model()`.
 - `pyCLINE.recovery_methods.data_preparation()`: Splits and normalizes that data for training, with many more features for the user to change the data.
 - `pyCLINE.recovery_methods.nn_training()`: Is the `pyTorch` implementation that sets up the model and trains it.

The `pyCLINE.model()` currently includes a set of different models: 
 - FitzHugh-Nagumo model
 - Bicubic model
 - Gene expression model
 - Glycolytic oscillation model
 - Goodwin model
 - Oregonator model
 - Lorenz system
 - Roessler system
 - Delay oscillator (self-inhibitory gene)

Some of the models are three-dimensional and can be used to further study the limitations of the method, when applied to higher dimensional systems

For a better understanding the method and a simpler implementation, we also provide `pyCLINE.example()` which contains full examples of how `pyCLINE` can be used.
Here, `pyCLINE.example()` can be used to generate prediction data for four systems: The FitzHugh-Nagumo model with time scale separation variable $\varepsilon=0.3$ (`FHN`), the Bicubic model (`Bicubic`), gene expression model (`GeneExpression`) and the delay oscillator model (`DelayOscillator`).

![The method CLINE explained by using Figure 1 from `@Prokop:2025`. In red the main modules of the `pyCLINE` package are shown \label{fig:method}](figures/introduction_manuscript_1.png)


# Acknowledgements


# References