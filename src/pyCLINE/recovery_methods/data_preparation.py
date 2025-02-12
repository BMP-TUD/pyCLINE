import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks


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
