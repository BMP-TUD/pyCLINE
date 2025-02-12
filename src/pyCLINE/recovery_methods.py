import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy.signal import find_peaks

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# -------------------  Data preparation -------------------

def normalize_variables(df,     # dataframe containing variables
                        vars,   # list of the variables to normalize
                        time,   # name of the time column
                        norm_method='minmax',
                        value_max=1,
                        value_min=0):  # name of the time column
    
    
    """The function conducts min-max normalization in the range '[-1:1]
    of the varibles from the list 'vars' in the dataframe 'df'

    return dataframe containing normalized variables 'df_norm'
    and the dataframe containing normalization coefficients per variable 'df_coef'

    Args:
        df (_type_): _description_
        value_max (int, optional): _description_. Defaults to 1.
        value_min (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_

    """
    # creating a new dataframe for normalized data and copying the time column
    df_norm = pd.DataFrame()
    df_norm[time] = df[time]
    # creating a dataframe to store the normalization coefficients (min and max of the raw variables)
    df_coef = pd.DataFrame(index=['min', 'max'])
    if norm_method != 'None':
        # normalizing variables
        for var in vars:
            if norm_method == 'minmax':
                df_coef[var] = [df[var].min(), df[var].max()]
                
                df_norm['norm {:}'.format(var)] = (df[var] - df_coef[var]['min'])*(value_max-value_min)/(df_coef[var]['max'] - df_coef[var]['min']) + value_min
            if norm_method == 'zscore':
                df_coef[var] = [df[var].mean(), df[var].std()]
                df_norm['norm {:}'.format(var)] = (df[var] - df_coef[var]['mean'])/(df_coef[var]['sd'])

    return df_norm, df_coef

def plot_optimal_thresholding(thresholds, nsamples, varsamples, optimal_threshold, idx, histogram):
    """_summary_

    Args:
        thresholds (_type_): _description_
        nsamples (_type_): _description_
        varsamples (_type_): _description_
        optimal_threshold (_type_): _description_
        idx (_type_): _description_
        histogram (_type_): _description_
    """  
    fig, axes = plt.subplots(1,2, figsize=(11,5))
    axes[0].plot(thresholds, nsamples, label='norm. sample size')
    axes[0].plot(thresholds, varsamples, label='norm. sampling SD', c='C1')
    axtwinx = axes[0].twinx()
    axtwinx.plot(thresholds, nsamples/varsamples, label='sample size/SD', c='C2')
    axtwinx.scatter(optimal_threshold, (nsamples/varsamples)[idx], c='r', s=50, label='optimal threshold, {:d}'.format(int(optimal_threshold)))

    axes[0].legend()
    axtwinx.legend()
    axes[0].set_xlabel('sampling threshold')
    axes[0].set_ylabel('norm sample size, sampling SD')
    axtwinx.set_ylabel('sample size/SD')

    hthresh = np.copy(histogram)
    hthresh[hthresh>optimal_threshold] = optimal_threshold

    axes[1].imshow(hthresh)

    plt.show()

def compute_optimal_threshold(df, vars, binx, biny, plot_thresholding=True):
    """_summary_

    Args:
        df (_type_): _description_
        vars (_type_): _description_
        binx (_type_): _description_
        biny (_type_): _description_
        plot_thresholding (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """    
    h, _, _ = np.histogram2d(df[vars[0]], df[vars[1]], bins=(binx, biny))

    nthresh = np.arange(np.min(h[h>0]),np.max(h[h>0]),50)
    nsamples = np.zeros(nthresh.shape[0])
    varsamples = np.zeros(nthresh.shape[0])

    for i in range(nthresh.shape[0]):

        hthresh = np.copy(h)
        hthresh[hthresh>nthresh[i]] = nthresh[i]
        nsamples[i] = np.sum(hthresh)
        varsamples[i] = np.std(hthresh[hthresh>0])

    if np.max(nsamples) != np.min(nsamples):
        nsamples = (nsamples - np.min(nsamples))/(np.max(nsamples) - np.min(nsamples)) + 1
        varsamples = (varsamples - np.min(varsamples))/(np.max(varsamples) - np.min(varsamples)) + 1

        idx = np.argmax(nsamples/varsamples)
        optimal_threshold = nthresh[idx]
    else:
        optimal_threshold = np.max(nthresh)

    if plot_thresholding:
        plot_optimal_thresholding(nthresh, nsamples, varsamples, optimal_threshold, idx, h)

    return int(optimal_threshold)

def uniform_sampling(df, threshold, input_vars, binx, biny):
    """_summary_

    Args:
        df (_type_): _description_
        threshold (_type_): _description_
        input_vars (_type_): _description_
        binx (_type_): _description_
        biny (_type_): _description_

    Returns:
        _type_: _description_
    """    
    df_uniform = pd.DataFrame()

    for i in range(0,binx.shape[0]):
        for j in range(0,biny.shape[0]):

            df_subsample = df[(df[input_vars[0]]>binx[i-1]) & (df[input_vars[0]]<binx[i]) &
                              (df[input_vars[1]]>biny[j-1]) & (df[input_vars[1]]<biny[j])].copy()
            if df_subsample.shape[0]>0:
                if df_subsample.shape[0]<=threshold:
                    df_uniform = pd.concat((df_uniform,df_subsample), ignore_index=True)
                else:
                    df_uniform = pd.concat((df_uniform,df_subsample[:threshold]), ignore_index=True)

    return df_uniform

# data preparation
def prepare_data(df,    # dataframe containing variables
                 vars,  # list of the variables to normalize
                 time,  # name of the time column
                 tmin=None, tmax=None, #   range of time to slice the data (optional)
                 scheme='newton_difference', # scheme for computing delayed variables
                 norm_method='minmax',
                 value_max=1,
                 value_min=0, 
                 normalize=True):
    """the function prepares the raw time series of the system variables
    for feeding into ML model. preparation includes following steps:
    (i)   data slicing in the indicated range [tmin:tmax], [:tmax], or [tmin:] (optional).
          if tmin and tmax are not provided, full data are processed;
    (ii)  min-max normalization of the variables from the list 'vars';
    (iii) coumputing a delayed [t-1] variable for each variable from the list.

    the function returns a prepared dataframe 'df_prepared'
    and the dataframe containing normalization coefficients per variable 'df_coef'

    Args:
        df (_type_): _description_
        tmax (_type_, optional): _description_. Defaults to None.
        value_max (int, optional): _description_. Defaults to 1.
        value_min (int, optional): _description_. Defaults to 0.
        normalize (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """    
    
    
    # slice the data in the range [tmin; tmax] if needed
    if not ((tmin is None) and (tmax is None)):
        if tmin is None:
            tmin = df[time].min()
        if tmax is None:
            tmax = df[time].max()
        df_slice = df[(df[time]>=tmin) & (df[time]<=tmax)].copy()

    # min-max normalization of each variable in the range [value_min; value_max]
    if normalize:
        df_norm, df_coef = normalize_variables(df_slice, vars=vars, time=time, norm_method=norm_method,value_max=value_max,value_min=value_min)
    else:
        df_norm = df_slice.copy()
        df_coef = pd.DataFrame(index=['min', 'max'])
        for var in vars:
            df_norm['norm {:}'.format(var)] = df_norm[var]
            df_coef[var] = [df[var].min(), df[var].max()]
    # computing delayed variables
    df_prepared = pd.DataFrame()
    
    if scheme == 'newton_difference':
        first_point = 1
    if scheme == 'two_point':
        first_point = 2   
    if scheme == 'five_point':
        first_point = 4
    if scheme == 'derivative':
        first_point=0
    df_prepared[time] = df_norm[time].iloc[first_point:]
    for var in vars:
        # df_prepared['norm {:}'.format(var)] = df_norm['norm {:}'.format(var)].to_numpy()[first_point:]
        if scheme=='newton_difference':
            df_prepared['norm {:}'.format(var)] = df_norm['norm {:}'.format(var)].to_numpy()[first_point:]
            df_prepared['norm {:}'.format(var)+'[t-1]'] = df_norm['norm {:}'.format(var)].to_numpy()[:-first_point]
        if scheme=='two_point':
            df_prepared['norm {:}'.format(var)] = df_norm['norm {:}'.format(var)].to_numpy()[first_point-1:-first_point+1]
            df_prepared['norm {:}'.format(var)+'[t+1]'] = df_norm['norm {:}'.format(var)].to_numpy()[first_point:]
            df_prepared['norm {:}'.format(var)+'[t-1]'] = df_norm['norm {:}'.format(var)].to_numpy()[first_point-2:-first_point]
        if scheme=='five_point':
            df_prepared['norm {:}'.format(var)] = df_norm['norm {:}'.format(var)].to_numpy()[first_point-2:-first_point+2]
            df_prepared['norm {:}'.format(var)+'[t+2]'] = df_norm['norm {:}'.format(var)].to_numpy()[first_point:]
            df_prepared['norm {:}'.format(var)+'[t+1]'] = df_norm['norm {:}'.format(var)].to_numpy()[first_point-1:-first_point+3]
            df_prepared['norm {:}'.format(var)+'[t-1]'] = df_norm['norm {:}'.format(var)].to_numpy()[first_point-3:-first_point+1]
            df_prepared['norm {:}'.format(var)+'[t-2]'] = df_norm['norm {:}'.format(var)].to_numpy()[:-first_point]
        if scheme=='derivative':
            dt=df[time][1]-df[time][0]
            df_prepared['norm {:}'.format(var)] = df_norm['norm {:}'.format(var)].to_numpy()[first_point:]
            # for var in vars:
            x = df_norm['norm {:}'.format(var)].to_numpy()[first_point:]
            x_dot = np.ones(x.shape[0])*np.NaN
            x_dot_forward =[(x[i+1]-x[i])/dt for i in range(x[:-1].shape[0])] #np.diff(x)/(dt) #(x[2:] - x[:-2])/(2*dt)
            
            x_dot_backward =[(x[i]-x[i-1])/dt for i in range(x[1:].shape[0])]

            x_dot_forward = np.array(x_dot_forward)
            x_dot_backward = np.array(x_dot_backward)
            x_dot= (x_dot_forward + x_dot_backward)/2
            # insert np.nan at the first position
            x_dot = np.insert(x_dot, -1, np.nan)
            x_dot[0]=np.nan
            # x_dot = np.insert(x_dot, 0, np.nan)
            
            df_prepared['d norm{:}'.format(var)+'/dt'] = x_dot
            # df_prepared['d norm{:}'.format(var)+'/dt'] = np.gradient(df_norm['norm {:}'.format(var)].to_numpy()[first_point:], dt)

    return df_prepared.dropna(), df_coef

def shuffle_and_split(df,               # dataframe containing prepared data for feeding into ML model
                      input_vars,       # the list of input variables
                      target_var,       # the list of target variable(s)
                      train_frac=0.7,   # fraction of training data (in the range [0,1], default is 0.7, i.e., 70%)
                      test_frac=0.15,   # fraction of testing data (in the range [0,1], default is 0.15, i.e., 15%)
                                        # remaining fraction is assinged as a validation data
                      optimal_thresholding=True,
                      plot_thresholding=True):
    """The function prepares training, testing, and validation sets from the prepared dataframe
    by random uniform shuffling and splitting the data according to the provided proportions

    the function returns respective training, testing, and validation datasets

    Args:
        df (_type_): _description_
        plot_thresholding (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """    

    # data shuffling
    df_shuffled = df.sample(frac = 1)

    # uniform data sampling in the phase space
    # optimal_thresholding = True

    if optimal_thresholding:
        binx = np.linspace(df_shuffled[input_vars[0]].min(), df_shuffled[input_vars[0]].max(), 11)
        biny = np.linspace(df_shuffled[target_var[0]].min(), df_shuffled[target_var[0]].max(), 11)

        if optimal_thresholding:
            optimal_threshold = compute_optimal_threshold(df_shuffled, [input_vars[0], target_var[0]], binx, biny, plot_thresholding)

        df_shuffled = uniform_sampling(df_shuffled, optimal_threshold, [input_vars[0], target_var[0]], binx, biny)
        df_shuffled = df_shuffled.sample(frac = 1)

    # computing the sizes of the training and testing datasets based of the provided fractions
    Ntrain = int(df_shuffled.shape[0]*train_frac)
    Ntest = int(df_shuffled.shape[0]*test_frac)

    # splitting the data
    input_train = df_shuffled[input_vars].iloc[:Ntrain]
    target_train = df_shuffled[target_var].iloc[:Ntrain]

    input_test = df_shuffled[input_vars].iloc[Ntrain:Ntrain+Ntest]
    target_test = df_shuffled[target_var].iloc[Ntrain:Ntrain+Ntest]

    input_val = df_shuffled[input_vars].iloc[Ntrain+Ntest:]
    target_val = df_shuffled[target_var].iloc[Ntrain+Ntest:]

    return input_train, target_train, input_test, target_test, input_val, target_val


# -------------------  ML modeling with PyTorch -------------------

class FFNN(nn.Module):   
    def __init__(self, Nin, Nout, Nlayers, Nnodes, activation):
        super(FFNN, self).__init__()
        layers = [nn.Linear(Nin, Nnodes), activation()]
        for _ in range(Nlayers - 1):
            layers.append(nn.Linear(Nnodes, Nnodes))
            layers.append(activation())
        layers.append(nn.Linear(Nnodes, Nout))
        # layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

def configure_FFNN_model(Nin,
                         Nout, 
                         Nlayers, 
                         Nnodes, 
                         activation=nn.Tanh,
                         optimizer_name='Adam',
                        #  optimizer=optim.Adam, 
                         lr=1e-4,
                         loss_fn=nn.MSELoss, 
                         summary=False):
    """_summary_

    Args:
        Nin (_type_): _description_
        Nout (_type_): _description_
        Nlayers (_type_): _description_
        Nnodes (_type_): _description_
        activation (_type_, optional): _description_. Defaults to nn.Tanh.
        optimizer_name (str, optional): _description_. Defaults to 'Adam'.
        lr (_type_, optional): _description_. Defaults to 1e-4.
        loss_fn (_type_, optional): _description_. Defaults to nn.MSELoss.
        summary (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """    
    model = FFNN(Nin, Nout, Nlayers, Nnodes, activation)
    model.apply(init_weights)
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=0.99)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-10)
    elif optimizer_name == 'Nadam':
        optimizer = optim.Nadam(model.parameters(), lr=lr)
    elif optimizer_name == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=lr)
    elif optimizer_name == 'Adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
    
    loss_fn = loss_fn()

    if summary:
        print(model)

    return model, optimizer, loss_fn

# Monitor gradients
def monitor_gradients(model):
    """_summary_

    Args:
        model (_type_): _description_

    Returns:
        _type_: _description_
    """    
    for name, param in model.named_parameters():
        if param.requires_grad:
            gradients=param.grad.norm()
            return gradients

def loss_function(input, target, nc_prediction, nullcline_guess, factor):
    """_summary_

    Args:
        input (_type_): _description_
        target (_type_): _description_
        nc_prediction (_type_): _description_
        nullcline_guess (_type_): _description_
        factor (_type_): _description_

    Returns:
        _type_: _description_
    """    
    mse_loss = nn.MSELoss()
    mse_loss_nc = nn.MSELoss()
    loss_train=mse_loss(input, target)
    loss_nc = mse_loss_nc(nc_prediction, nullcline_guess)
    return loss_nc+factor*loss_train

def train_FFNN_model(model,                 # configured FFNN model
                     optimizer,             # optimizer
                     loss_fn,               # loss function
                     input_train,           # input training data
                     target_train,          # target training data
                     input_test,            # input testing data
                     target_test,           # target testing data
                     validation_data,       # validation data
                     epochs=200,            # number of epochs
                     batch_size=64,         # batch size
                     plot_loss=True,        # plot the loss
                     device='cpu',          # device to run the model
                     use_progressbar=True,   # use progress bar
                     save_evolution=True,
                     loss_target='limit_cycle',
                     nullcline_guess=None, 
                     factor=1.0, 
                     method=None,
                     minimal_value=0,
                     maximal_value=1
                     ):
    """_summary_

    Args:
        model (_type_): _description_
        loss_target (str, optional): _description_. Defaults to 'limit_cycle'.
        nullcline_guess (_type_, optional): _description_. Defaults to None.
        factor (float, optional): _description_. Defaults to 1.0.
        method (_type_, optional): _description_. Defaults to None.
        minimal_value (int, optional): _description_. Defaults to 0.
        maximal_value (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    # Move model to the specified device
    model.to(device)
    if loss_target=='limit_cycle':
        train_dataset = TensorDataset(torch.tensor(input_train.values, dtype=torch.float32), torch.tensor(target_train.values, dtype=torch.float32))
        test_dataset = TensorDataset(torch.tensor(input_test.values, dtype=torch.float32), torch.tensor(target_test.values, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(validation_data[0].values, dtype=torch.float32), torch.tensor(validation_data[1].values, dtype=torch.float32))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    if loss_target=='nullcline_guess':
        
        train_dataset = TensorDataset(torch.tensor(input_train.values, dtype=torch.float32), torch.tensor(target_train.values, dtype=torch.float32), torch.tensor(np.array([nullcline_guess]*input_train.shape[0]), dtype=torch.float32))
        test_dataset = TensorDataset(torch.tensor(input_test.values, dtype=torch.float32), torch.tensor(target_test.values, dtype=torch.float32), torch.tensor(np.array([nullcline_guess]*input_test.shape[0]), dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(validation_data[0].values, dtype=torch.float32), torch.tensor(validation_data[1].values, dtype=torch.float32), torch.tensor(([nullcline_guess]*validation_data[0].shape[0]), dtype=torch.float32))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_losses = []
    val_losses = []
    predictions=[]
    lc_predictions=[]
    gradients=[]
        # if use_progressbar:
    #     progressbar=tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}')

    for epoch in range(epochs):
        # with tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}') as progressbar if use_progressbar else enumerate(train_loader:
            if use_progressbar:
                progressbar=tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}')
            else:
                progressbar=enumerate(train_loader)
            model.train()
            running_loss = 0.0
            for batch_idx, data in progressbar:
                # Move inputs and targets to the specified device
                if loss_target=='limit_cycle':
                    inputs, targets = data
                    inputs, targets = inputs.to(device), targets.to(device)
                if loss_target=='nullcline_guess':
                    inputs, targets, nullcline_guess = data
                    inputs, targets, nullcline_guess = inputs.to(device), targets.to(device), nullcline_guess.to(device)
                # inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                if loss_target=='limit_cycle':
                    loss = loss_fn(outputs, targets)

                if loss_target=='nullcline_guess':
                    input_null = np.zeros((nullcline_guess.shape[1], model.model[0].in_features))
                    for i in range(model.model[0].in_features):
                        input_null[:,i] = np.linspace(0, 1,nullcline_guess.shape[1])
                    input_null = torch.tensor(input_null, dtype=torch.float32).to(device)

                    nc_prediction = model(input_null)
                    
                    # _, nc_prediction = nullcline_prediction(model, Nsteps=nullcline_guess.shape[1])
                    # Ensure the tensors require gradients
                    # nc_prediction = torch.tensor(np.array([nc_prediction]*nullcline_guess.shape[0]), dtype=torch.float32, requires_grad=True)
                    
                    loss = loss_function(outputs, targets, nc_prediction[:,0], nullcline_guess[0,:], factor)
                # loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            
            # gradients.append(monitor_gradients(model))

            train_loss = running_loss / len(train_loader.dataset)
            train_losses.append(train_loss)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data in val_loader:
                    
                    if loss_target=='limit_cycle':
                        inputs, targets = data
                        inputs, targets = inputs.to(device), targets.to(device)
                    if loss_target=='nullcline_guess':
                        inputs, targets, nullcline_guess = data
                        inputs, targets, nullcline_guess = inputs.to(device), targets.to(device), nullcline_guess.to(device)

                    outputs = model(inputs)
                    if loss_target=='limit_cycle':
                        loss = loss_fn(outputs, targets)

                    if loss_target=='nullcline_guess':
                        input_null = np.zeros((nullcline_guess.shape[1], model.model[0].in_features))
                        for i in range(model.model[0].in_features):
                            input_null[:,i] = np.linspace(0, 1,nullcline_guess.shape[1])
                        input_null = torch.tensor(input_null, dtype=torch.float32).to(device)

                        nc_prediction = model(input_null)
                        
                        # _, nc_prediction = nullcline_prediction(model, Nsteps=nullcline_guess.shape[1])
                        # Ensure the tensors require gradients
                        # nc_prediction = torch.tensor(np.array([nc_prediction]*nullcline_guess.shape[0]), dtype=torch.float32, requires_grad=True)
                        
                        loss = loss_function(outputs, targets, nc_prediction[:,0], nullcline_guess[0,:], factor)
                    val_loss += loss.item() * inputs.size(0)

            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)

            if save_evolution:# and (epoch+1)%10==0:
                _ , prediction_null = nullcline_prediction(model, Nsteps=500, method=method, min_val=minimal_value, max_val=maximal_value)
                predictions.append(prediction_null)
                # model.eval()
                input_prediction=torch.tensor(input_train.values, dtype=torch.float32)
                with torch.no_grad():
                    output_prediction = model(input_prediction).cpu().numpy()
                lc_predictions.append(output_prediction)
            # Update the progress bar description with the current loss
            if use_progressbar:
                progressbar.set_description(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {train_loss:.4f}')
                progressbar.refresh()  
            # if plot_loss:
            #     print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    if plot_loss:
        # plt.plot(gradients, label='Gradients')
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.yscale('log')
        plt.legend()
        plt.show()

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for data in test_loader:
            if loss_target=='limit_cycle':
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)
            if loss_target=='nullcline_guess':
                inputs, targets, nullcline_guess = data
                inputs, targets, nullcline_guess = inputs.to(device), targets.to(device), nullcline_guess.to(device)

            outputs = model(inputs)
            if loss_target=='limit_cycle':
                loss = loss_fn(outputs, targets)

            if loss_target=='nullcline_guess':
                input_null = np.zeros((nullcline_guess.shape[1], model.model[0].in_features))
                for i in range(model.model[0].in_features):
                    input_null[:,i] = np.linspace(0, 1,nullcline_guess.shape[1])
                input_null = torch.tensor(input_null, dtype=torch.float32).to(device)

                nc_prediction = model(input_null)
                
                # _, nc_prediction = nullcline_prediction(model, Nsteps=nullcline_guess.shape[1])
                # Ensure the tensors require gradients
                # nc_prediction = torch.tensor(np.array([nc_prediction]*nullcline_guess.shape[0]), dtype=torch.float32, requires_grad=True)
                # print(nullcline_guess.shape, nc_prediction.shape)
                loss = loss_function(outputs, targets, nc_prediction[:,0], nullcline_guess[0,:], factor)
                # loss= loss_fn(outputs, targets)+loss_fn(nc_prediction, nullcline_guess)
            test_loss += loss.item() * inputs.size(0)

    test_loss /= len(test_loader.dataset)
    # print(f'Test Loss: {test_loss:.4f}')

    return train_losses, val_losses, test_loss, predictions, lc_predictions

def nullcline_prediction(model, Nsteps, device='cpu', method=None, min_val=0, max_val=1):

    input_null = np.zeros((Nsteps, model.model[0].in_features))
    if method=='derivative':
        for i in range(model.model[0].in_features-1):
            input_null[:,i] = np.linspace(min_val, max_val, Nsteps)
    else:
        for i in range(model.model[0].in_features):
            input_null[:,i] = np.linspace(min_val, max_val,Nsteps)
    # input_null = np.repeat(input_null, Nin, axis=1)  # Repeat to match the number of input features
    input_null = torch.tensor(input_null, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        prediction_null = model(input_null).cpu().numpy()
    return input_null.cpu().numpy(), prediction_null.reshape(Nsteps,)

def compute_nullcline(fnull,
                      p,
                      xvar,
                      yvar, 
                      Nsteps,
                      df_coef, 
                      value_max=1,
                      value_min=0, 
                      normalize=True):
    
    xnull = np.linspace(df_coef[xvar]['min'],df_coef[xvar]['max'], Nsteps)
    
    ynull = fnull(xnull)
    if normalize:
        ynull = (ynull - df_coef[yvar]['min'])*(value_max-value_min)/(df_coef[yvar]['max'] - df_coef[yvar]['min'])+value_min

    return xnull, ynull

def normalize_adjusted(x, df_coef, var, min=-1, max=1):
    return (x - df_coef[var].min())*(max-min) / (df_coef[var].max() - df_coef[var].min())+min


#### General functions
def calculate_period(x_train, t_train):
    """
    Calculate the period of an oscillation.

    Parameters
    ----------
    x_train : np.array
        Time series of data.
    t_train : np.array
        Time of the time series.

    Returns
    -------
    peaks_u : np.array
        Size of peaks detected in the time series.
    peaks_t : np.array
        Time points where peaks occur in the data set.
    period : float
        Period of Time Series.
    peaks[0]: list
        Array of all indicies of local maxima
    
    """
    if len(x_train.shape)<2:
        peaks=find_peaks(x_train)
        peaks_t=t_train[peaks[0]]
        peaks_u=x_train[peaks[0]]
    elif x_train.shape[1]==2 or x_train.shape[1]==3:
        peaks=find_peaks(x_train[:,0])
        peaks_t=t_train[peaks[0]]
        peaks_u=x_train[peaks[0],0]
    elif x_train.shape[1]==5:
        peaks=find_peaks(x_train[:,1])
        peaks_t=t_train[peaks[0]]
        peaks_u=x_train[peaks[0],1]

    period_temp=0
    for i in range(len(peaks_t)-1):
        period_temp=period_temp+(peaks_t[i+1]-peaks_t[i])
    if len(peaks_t)>1:
        subtract=1
    else: subtract=0
    period=period_temp/(len(peaks_t)-subtract)
    return peaks_u, peaks_t, period, peaks[0]
