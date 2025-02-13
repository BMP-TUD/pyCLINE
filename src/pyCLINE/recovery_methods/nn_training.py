
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class FFNN(nn.Module):   
    """
    Feedforward Neural Network (FFNN) class.

    Args:
        nn (torch.nn module): PyTorch neural network module
    """
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
    """
    Initialize the weights of the neural network.

    Args:
        m (torch model): PyTorch model
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

def configure_FFNN_model(Nin, Nout, Nlayers, Nnodes,  activation=nn.SiLU, optimizer_name='Adam', lr=1e-4,
                         loss_fn=nn.MSELoss, summary=False):
    """
    Configure the Feedforward Neural Network (FFNN) model.

    Args:
        Nin (int): Number of input features
        Nout (int): Number of output features
        Nlayers (int): Number of layers
        Nnodes (int): Number of nodes
        activation (torch neural network activation function module, optional): Activation function of neural network training. Defaults to nn.SiLU.
        optimizer_name (str, optional): Optimizer for model training. Defaults to 'Adam'.
        lr (float, optional): Learning rate for training. Defaults to 1e-4.
        loss_fn (torch loss function module, optional): Loss function for training. Defaults to nn.MSELoss.
        summary (bool, optional): If model summary should be generated. Defaults to False.

    Returns:
        model (torch model): setup FFNN model
        optimizer (torch optimizer): optimizer for training
        loss_fn (torch loss function): loss function for training
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
    """
    Monitor the gradients of the model.

    Args:
        model (torch model feature): torch model

    Returns:
        gradients (float): gradients of the model
    """    
    for name, param in model.named_parameters():
        if param.requires_grad:
            gradients=param.grad.norm()
            return gradients

def loss_function(input, target, nc_prediction, nullcline_guess, factor):
    """
    Loss function for the neural network model.
    Args:
        input (torch tensor): input data
        target (torch tensor): target data
        nc_prediction (torch tensor): nullcline prediction data
        nullcline_guess (torch tensor): nullcline guess data
        factor (float): penalty factor for loss function

    Returns:
        loss (float): loss function
    """    
    mse_loss = nn.MSELoss()
    mse_loss_nc = nn.MSELoss()
    loss_train=mse_loss(input, target)
    loss_nc = mse_loss_nc(nc_prediction, nullcline_guess)
    return loss_nc+factor*loss_train

def train_FFNN_model(model, optimizer, loss_fn, input_train, target_train, input_test, target_test, validation_data,
                     epochs=200, batch_size=64, plot_loss=True, device='cpu', use_progressbar=True, save_evolution=True,
                     loss_target='limit_cycle', nullcline_guess=None, factor=1.0, method=None, minimal_value=0.0, maximal_value=1.0):
    """
    Train the Feedforward Neural Network (FFNN) model.

    Args:
        model (torch model): configured FFNN model
        optimizer (torch optimizer): optimizer
        loss_fn (torch loss function): loss function
        input_train (pandas dataframe): input training data
        target_train (pandas dataframe): target training data
        input_test (pandas dataframe): input testing data
        target_test (pandas dataframe): target testing data
        validation_data (pandas dataframe): validation data
        epochs (int, optional): number of epochs. Defaults to 200.
        batch_size (int, optional): batch size. Defaults to 64.
        plot_loss (bool, optional): plot the loss. Defaults to True.
        device (str, optional): device to train the model on. For 'cuda', necessary GPU and CUDA Toolkit are required. Defaults to 'cpu'.
        use_progressbar (bool, optional): use progress bar when running the training. Defaults to True.
        save_evolution (bool, optional): save the evolution of limit cycle and nullcline predictions. Defaults to True.
        loss_target (str, optional): Decide if the target is not limit cycle, but the inital nullcline seed. Defaults to 'limit_cycle'.
        nullcline_guess (torch tensor, optional): Torch tensor containing inital nullcline guess. Defaults to None.
        factor (float, optional): The penalty factor for nullcline guess loss evaluation. Defaults to 1.0.
        method (str, optional): Method used for prediction, can be derivative or delayed variables. Defaults to None.
        minimal_value (float, optional): Minimal value of minmax normalization for inbetween prediction. Defaults to 0.0.
        maximal_value (int, optional): Maximal value of minmax normalization for inbetween prediction. Defaults to 1.0.

    Returns:
        train_losses (list): training losses
        val_losses (list): validation losses
        test_loss (float): testing loss
        predictions (list): predictions of nullcline structure over all epochs
        lc_predictions (list): limit cycle predictions over all epochs
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

    for epoch in range(epochs):
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
                    loss = loss_function(outputs, targets, nc_prediction[:,0], nullcline_guess[0,:], factor)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            
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
                        loss = loss_function(outputs, targets, nc_prediction[:,0], nullcline_guess[0,:], factor)
                    val_loss += loss.item() * inputs.size(0)

            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)

            if save_evolution:
                _ , prediction_null = nullcline_prediction(model, Nsteps=500, method=method, min_val=minimal_value, max_val=maximal_value)
                predictions.append(prediction_null)
                input_prediction=torch.tensor(input_train.values, dtype=torch.float32)
                with torch.no_grad():
                    output_prediction = model(input_prediction).cpu().numpy()
                lc_predictions.append(output_prediction)
            if use_progressbar:
                progressbar.set_description(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {train_loss:.4f}')
                progressbar.refresh()  

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
                loss = loss_function(outputs, targets, nc_prediction[:,0], nullcline_guess[0,:], factor)
            test_loss += loss.item() * inputs.size(0)

    test_loss /= len(test_loader.dataset)

    return train_losses, val_losses, test_loss, predictions, lc_predictions

def nullcline_prediction(model, Nsteps, device='cpu', method=None,min_val=0.0, max_val=1.0):
    """
    Predict the nullcline of the model.

    Args:
        model (torch model): FFNN model
        Nsteps (int): Number of steps for discretization of input variable for nullcline prediction
        device (str, optional): device to train the model on. For 'cuda', necessary GPU and CUDA Toolkit are required. Defaults to 'cpu'.
        method (str, optional): Method used for prediction, can be derivative or delayed variables. Defaults to None.
        min_val (float, optional): Minimum value for minmax normalization. Defaults to 0.0.  
        max_val (float, optional): Maximum value for minmax normalization. Defaults to 1.0.

    Returns:
        input_null (numpy array): input variable for nullcline prediction
        prediction_null (numpy array): predicted nullcline
    """
    input_null = np.zeros((Nsteps, model.model[0].in_features))
    if method=='derivative':
        for i in range(model.model[0].in_features-1):
            input_null[:,i] = np.linspace(min_val, max_val, Nsteps)
    else:
        for i in range(model.model[0].in_features):
            input_null[:,i] = np.linspace(min_val, max_val,Nsteps)

    input_null = torch.tensor(input_null, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        prediction_null = model(input_null).cpu().numpy()
    return input_null.cpu().numpy(), prediction_null.reshape(Nsteps,)

def compute_nullcline(fnull, xvar, yvar, Nsteps, df_coef, value_min=0.0, value_max=1.0, normalize=True):
    """
    Compute the nullcline of the model from the ground true function.

    Args:
        fnull (function): ground true function
        xvar (str): x variable for nullcline computation
        yvar (str): y variable for nullcline computation
        Nsteps (int): Number of steps for discretization of input variable for nullcline prediction
        df_coef (pandas dataframe): dataframe containing the coefficients of the model
        value_min (float, optional):  Minimum value for minmax normalization. Defaults to 0.0.
        value_max (float, optional):  Maximum value for minmax normalization.. Defaults to 1.0.
        normalize (bool, optional): If values should be minmax normalized. Defaults to True.

    Returns:
        xnull (list): x variable for nullcline prediction
        ynull (list): predicted nullcline
    """
    xnull = np.linspace(df_coef[xvar]['min'],df_coef[xvar]['max'], Nsteps)
    ynull = fnull(xnull)
    if normalize:
        ynull = (ynull - df_coef[yvar]['min'])*(value_max-value_min)/(df_coef[yvar]['max'] - df_coef[yvar]['min'])+value_min

    return xnull, ynull
