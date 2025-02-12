
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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
