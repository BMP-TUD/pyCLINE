---
title: 'pyCLINE: A Python package using the CLINE method for discovery of hidden nullcline structures in oscillatory dynamical systems'
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
    affiliation: 1
  - name: Nikit Frolov 
    affiliation: 1
    orcid: 0000-0002-2788-1907
  - name: Lendert Gelens
    affiliation: 1
    orcid: 0000-0001-7290-9561
affiliations:
 - name: Laboratory of Dynamics in Biological Systems, Department of Cellular and Mollecular Medicine, KU Leuven
   index: 1
date: 21 March 2025
bibliography: paper.bib
---

# Summary

Dynamical processes in physics, biology, chemistry, and engineering—such as planetary motion, climate variability, and cell cycle oscillations—are crucial to understanding complex systems. 
Traditionally, mathematical models describing these systems rely on differential equations derived from empirical data using established modeling principles and scientific intuition. 
However, the increasing availability of high-dimensional, complex datasets has rendered classical model derivation increasingly challenging.

As a result, data-driven or machine learning methods emerged that are able to handle high-dimensional data sets. 
However existing methods either are limited by data quality or the interpretability of their results. 
Thus we developed the **CLINE** (**C**omputational **L**earning and **I**nference of **N**ullclin**E**s) [@Prokop2025] and introduce its Python implementation `pyCLINE` that allows to extract static to identify static phase-space structures without prior knowledge directly from time series data.  

# Statement of need

Machine learning and data-driven approaches have revolutionized the study of dynamical systems. Two primary methodologies exist:

- **Black-box methods** (e.g., neural networks) approximate system behavior but lack interpretability regarding underlying mechanisms.
- **White-box methods** derive symbolic differential equations directly from data but require high-quality datasets to ensure accuracy [@Prokop2024].

To bridge this gap, **grey-box methods** integrate the strengths of both approaches, handling large, structured datasets while preserving interpretability. Examples include Physics-Informed Neural Networks (PINNs) [@Karniadakis2021], Biology-Informed Neural Networks (BINNs) [@Lagergren2020], and Universal Differential Equations [@Rackauckas2020]. However, most of these methods focus on forecasting rather than extracting fundamental structural properties of dynamical systems.

To address this limitation, our method **CLINE** is able to extract static phase-space features, specifically the structure of nullclines, from time series data without forecasting.  
Understanding nullcline structure of a dynamical system provides several key benefits [@Prokop2024b]:

- **Comprehensive System Characterization:** Nullclines fully describe the system’s steady-state behavior and provide richer insights than time series data alone.
- **Reduced Complexity for Symbolic Model Identification:** Once nullcline structures are identified, symbolic equations can be inferred using sparse regression techniques, such as sparse identification of nonlinear dynamics (SINDy) [@Brunton2016] or symbolic regression (SR) [@Schmidt2009], with significantly lower computational complexity compared to direct time-series-based approaches.
- **Bias Reduction through Model-Free Inference:** Unlike traditional white-box methods, CLINE does not rely on predefined candidate terms (e.g., library-based functions), minimizing biases in model formulation and increasing adaptability to diverse systems.

`pyCLINE` is a Python package that allows one to easily set up and use the CLINE method as explained and shown in @Prokop2025. It is based on the Python Torch implementation `pyTorch` [@Paszke2019] and enables rapid identification of nullcline structures from simulated or measured time series data. 
The implementation of `pyCLINE` can generate exemple data sets from scratch, correctly prepare data for training and set up the feed forward neural network for training. 

`pyCLINE` was designed to be used by researchers experienced with the use of machine learning or laymen that are interested in applying the method to either different models or measured data. 
This allows for simple and fast implementation in many fields that are interested in discovering nullcline structures from measured data, that can help develop novel or confirm existing models of dynamical (oscillatory) systems.

## Methodology

The main aspects of the CLINE method are explained in @Prokop2025, nevertheless we provide a brief explanation of the method. 
In order to identify nullclines for a set of ordinary differential equations (ODEs) with system variables $u$ and $v$, we have to set the derivative to 0: 

$$u_t = f(u,v) \rightarrow u_t = f(u,v)=0$$    
$$v_t = g(u,v) \rightarrow v_t = g(u,v)=0$$

The functions of $f$ and $g$ are not known *a priori*.
However, to learn the functions we can reformulate the nullcline equations to:

$$u = f^{-1}(v,u_t)\text{ or } v = f^{-1}(u,u_t)$$
$$u = g^{-1}(v,v_t)\text{ or } v = g^{-1}(u,v_t)$$

Now we have to learn the inverse functions $f^{-1}$ and $g^{-1}$ which describe the relationship between the measured variables $u$ and $v$ with additional derivative information $u_t$ or $v_t$
As such, the target functions can be expressed as a feed-forward neural network with e.g. inputs $u$ and $u_t$, to learn $v$. 

After training, we can provide a set of $u$ together with $u_t=0$ (requirement for a nullcline) as inputs and learn the corresponding values of $v$ that describe $u_t = f(u,v)=0$.
As a result, we learn the structure of a nullcline in the phase space $u,v$, to which other white-box methods can be applied to learn the symbolic equations, yet on a decisively simpler optimization problem then that on time series data.

# Usage

The `pyCLINE` package can be downloaded and installed using `pip`:

    pip install pyCLINE

The `pyCLINE` package includes three main modules (see \autoref{fig:method}): 

 - `pyCLINE.generate_data()`: This module generates data which has been used in @Prokop2025 along with many additional models that can be found under `pyCLINE.model()`.
 - `pyCLINE.recovery_methods.data_preparation()`: Splits and normalizes that data for training, with many more features for the user to change the data.
 - `pyCLINE.recovery_methods.nn_training()`: The `pyTorch` implementation that sets up the model and trains it.

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

Some of the models are three-dimensional and can be used to further study the limitations of the method, when applied to higher dimensional systems.

To demonstrate the method, we also provide `pyCLINE.example()` which contains full examples of how `pyCLINE` can be used.
Here, `pyCLINE.example()` can be used to generate prediction data for four systems: The FitzHugh-Nagumo model with time scale separation variable $\varepsilon=0.3$ (`FHN`), the Bicubic model (`Bicubic`), gene expression model (`GeneExpression`) and the delay oscillator model (`DelayOscillator`).

![The method CLINE explained by using Figure 1 from @Prokop2025. In red the main modules of the `pyCLINE` package are shown. \label{fig:method}](figures/introduction_manuscript_1.png)


# References
