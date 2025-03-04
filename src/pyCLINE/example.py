import sys 
import os
from . import generate_data
import pandas as pd
import matplotlib.pyplot as plt
from . import recovery_methods
from . import model
import torch.nn as nn
import numpy
import torch

class ExampleCallable:
    def __call__(self, example_model, plot=True):
        example(example_model, plot)

sys.modules[__name__]= ExampleCallable()

def example(example_model, plot):
    """
    This function runs multiple examples, depending on the choice of the example model. 
    It should be used as a guideline how to run pyCLINE for synthetic data. 
    For using pyCLINE on real data, the user should adapt the data source accordingly.

    Args:
        example_model (str): Selection of model to run the example on. Chose from 'FHN', 'Bicubic', 'GeneExpression', 'DelayOscillator'.
        plot (bool): If True, the function will plot the data and the predictions.

    Raises:
        ValueError: In case no example model string is provided.
    """

    #provide name of model to run example
    if example_model is None:
        raise ValueError("Error: example_model is None: please provide a model name (FHN, Bicubic, GeneExpression, DelayOscillator).")
    
    # check if data already exists
    path='data/synthetic_data/'
    if example_model=='FHN': 
        fname='FHN_eps=0.3_a=0.0.csv'
    else: 
        fname=example_model+'.csv'
    
    print('Running example for model: '+example_model)
    print('Step 1: Load or generate data')
    # generate data if it does not exist
    if os.path.exists(path+fname):
        print('Data already exists: '+path+fname)
        df = pd.read_csv(path+fname)

    else:
        print('No data saved: generating data for model: '+example_model)
        getattr(generate_data, example_model)()
        print('Data generated saved: '+path+fname)
        df = pd.read_csv(path+fname)
    
    # extracting 1 time series and plotting
    if example_model!='DelayOscillator':
        df_sim = df[(df['sim']==1)].copy()
    else:
        df_sim=df.copy()
    df_sim.reset_index(drop=True, inplace=True)

    if example_model=='DelayOscillator':
        tau=10 # 12,20,35
        v_data=df_sim['u'][df_sim['time']>=tau].values
        df_sim=df_sim[df_sim['time']<=df_sim['time'].max() - tau]
        df_sim['v']=v_data

    if plot:
        fig,ax = plt.subplots(1,1,figsize=(5,3))
        ax.plot(df_sim['time'], df_sim['u'], label='u')
        ax.plot(df_sim['time'], df_sim['v'], label='v')
        ax.set_xlabel('Time')
        ax.set_ylabel('Amount in arbs. units')
        plt.show()
    
    # prepare data for training
    # tmin: minimum time point to consider to avoid transient behavior
    # val_min, val_max: min and max values of the min max normalization
    print('Step 2: Prepare data for training')
    if example_model=='FHN':
        tmin, val_min, val_max=3, -1.0, 1.0
    if example_model=='Bicubic':
        tmin, val_min, val_max=10, 0.0, 1.0 
    if example_model=='GeneExpression':
        tmin, val_min, val_max=10, 0.0, 1.0
    if example_model=='DelayOscillator':
        tmin, val_min, val_max=100,0.0,1.0


    # normalize data and create additional derivative or delayed variables
    df_sim, df_coef = recovery_methods.data_preparation.prepare_data(df_sim, vars=['u', 'v'], time='time', tmin=tmin, scheme='derivative', value_min=val_min, value_max=val_max)

    # define input and target variables (can be changed as needed)
    input_vars, target_vars = ['norm u', 'd normu/dt'], ['norm v']

    # schuffle and split training, test and validation data
    # optimal_thresholding: if True, the amount of samples within the phase space is evenly distributed
    input_train, target_train, input_test, target_test, input_val, target_val = recovery_methods.data_preparation.shuffle_and_split(df_sim, input_vars = input_vars,
                                                                                                                    target_var = target_vars,
                                                                                                                    optimal_thresholding=False)
    
    # set up the model
    # Nin: number of input variables
    # Nout: number of output variables
    # Nlayers: number of layers
    # Nnodes: number of nodes per layer
    # summary: if True, the model summary is printed
    # lr: learning rate
    # activation: activation function can be chosen as needed
    print('Step 3: Set up the model')
    nn_model,  optimizer, loss_fn = recovery_methods.nn_training.configure_FFNN_model(Nin=len(input_vars), Nout=len(target_vars),
                                                                                   Nlayers=3, Nnodes=64, summary=True, lr=1e-4,
                                                                                   activation=nn.SiLU)
    
    # train the model
    print('Step 4: Train the model')
    training_loss, val_loss, test_loss, predictions_evolution, lc_predictions = recovery_methods.nn_training.train_FFNN_model(model=nn_model,
                                                                                                                            optimizer=optimizer, loss_fn=loss_fn,
                                                                                                                            input_train=input_train,
                                                                                                                            target_train=target_train,input_test=input_test, 
                                                                                                                            target_test=target_test, 
                                                                                                                            validation_data=(input_val, target_val),
                                                                                                                            epochs=3000, batch_size=64, device='cpu',save_evolution=True, 
                                                                                                                            method='derivative', minimal_value=val_min,maximal_value=val_max)

    # save model and predictions
    print('Step 5: Save the generated predictions and model')
    if not os.path.exists(f'results/{example_model}'):
        os.makedirs(f'results/{example_model}')

    torch.save(nn_model.state_dict(), f'results/{example_model}/model.pth')
    numpy.save(f'results/{example_model}/predictions.npy', predictions_evolution)
    numpy.save(f'results/{example_model}/lc_predictions.npy', lc_predictions)
    numpy.save(f'results/{example_model}/training_loss.npy', training_loss)
    numpy.save(f'results/{example_model}/val_loss.npy', val_loss)
    numpy.save(f'results/{example_model}/test_loss.npy', test_loss)

    print('Example completed: model and predictions saved in results/'+example_model)

    # plot the predictions
    if plot:
        print('Step 6: Plot the predictions')
        fig,ax = plt.subplots(1,1,figsize=(5,3))
        ax.scatter(df_sim['norm u'], df_sim['norm v'], label='GT LC', c='silver')
        ax.scatter(input_train['norm u'], lc_predictions[-1,:,0],
                c='C2', label='Pred. LC', s=2 )
        
        # compute nullcline
        u = numpy.linspace(df_coef['u'].min(), df_coef['u'].max(), predictions_evolution.shape[1])
        sim_model = getattr(model, example_model)()
        gt_nullcline=sim_model.vnull(u)
        
        norm_u=recovery_methods.data_preparation.normalize_adjusted(u, df_coef,'u', 
                                                                    min=val_min, max=val_max)
        norm_gt_nullcline=recovery_methods.data_preparation.normalize_adjusted(gt_nullcline, df_coef,'v', 
                                                                               min=val_min, max=val_max)
        
        ax.plot(norm_u,norm_gt_nullcline, label='GT NC', c='k')
        ax.plot(norm_u,predictions_evolution[-1,:], label='Pred. NC', c='C1')
        ax.set_xlabel('u')
        ax.set_ylabel('v')
        ax.legend()
        plt.show()
