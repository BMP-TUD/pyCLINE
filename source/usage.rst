Example of use
==============

As mentioned before, in order to run several examples, the module ``example`` can be called

.. code-block:: python

    import pyCLINE
    pyCLINE.example(*example name*)

where example names are ``FHN`` (FitzHugh-Nagumo model), ``Bicubic`` (Bicubic model), ``GeneExpression`` (Gene expression) or ``DelayOscillator`` (Delay oscillator model) which are introduced in the manuscript. 

Additionally, below you can find the example for a the use of the FitzHugh-Nagumo (FHN) model without the use of  ``pyCLINE.example``.
After installing the package, we import ``pyCLINE`` as well as ``torch.nn`` to be able to configure the activation function for the neural network training. 
Additionally, we load ``pandas`` to be able to load generated test data sets.

.. code-block:: python

    import pyCLINE
    import torch.nn as nn
    import pandas

Firstly, we start by generating the FHN data set, which we then can load as a Pandas Dataframe back into the file from the created data directory:

.. code-block:: python
    
    pyCLINE.generate_data.FHN(dt=0.1, N=1000000, epsilons=[0.3], n_intiaL_conditions=1)
    df = pd.read_csv('data/synthetic/FHN_eps=0.3_a=0.0)


This prepared Dataframe consists of many simulations from randomly selected initial conditions, but we just want to use a single one and reset the index of the dataframe. 

.. code-block:: python

    df_sim = df[(df['sim']==1)].copy()
    df_sim.reset_index(drop=True, inplace=True)

This step can be skipped when using a single simulation with your data when you only have a single time series.
After this we can prepare the data for training, where we declare the column names with a set of parameters used to normalize the data:

.. code-block:: python

    df_sim, df_coef = pyCLINE.recovery_methods.data_preparation.prepare_data(df_sim, vars=['u', 'v'], time='time', tmin=10, scheme='derivative', value_min=0.0, value_max=1.0)

We then can define the variables that will be used as input and output/target variables of the neural network, and split the datasets into training, test and validation:

.. code-block:: python

    input_train, target_train, input_test, target_test, input_val, target_val = pyCLINE.recovery_methods.data_preparation.shuffle_and_split(df_sim, input_vars = input_vars, target_var = target_vars, optimal_thresholding=False)

With the prepared data, we can set up the model and train it:

.. code-block:: python

    #set up
    nn_model,  optimizer, loss_fn = recovery_methods.nn_training.configure_FFNN_model(Nin=len(input_vars), Nout=len(target_vars),Nlayers=3, Nnodes=64, summary=True, lr=1e-4, activation=nn.SiLU)

    #training
    training_loss, val_loss, test_loss, predictions_evolution, lc_predictions = recovery_methods.nn_training.train_FFNN_model(model=nn_model, optimizer=optimizer, loss_fn=loss_fn, input_train=input_train,target_train=target_train,input_test=input_test, target_test=target_test, validation_data=(input_val, target_val), epochs=3000, batch_size=64, device='cpu',save_evolution=True,method='derivative', minimal_value=val_min,maximal_value=val_max)

The result of the training are the losses and the predictions of the limit cycle (``lc_predictions``) and nullcline predictions (``predictions_evolution``) over the set amount of epochs, which can be used to visualize the outcome of the nullcline predictions. 