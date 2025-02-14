import numpy as np
from . import model

def FHN(dt=0.1, N=1000000, epsilons=[0.3], n_intiaL_conditions=1):
    """
    Generate data using the FitzHugh-Nagumo model (see src/pyCLINE/model.py)
    with different time scale separation, as used in Prokop, Billen, Frolov, Gelens (2025).

    Args:
        dt (float): Time step. Defaults to 0.1.
        N (int): Number of time steps. Defaults to 1000000.
        epsilons (list): List of time scale separations. Defaults to [0.3].
    
    Returns:
        None
    """
    u = np.zeros([2,N])
    for i_eps, eps in enumerate(epsilons):
        u[:,0] = [0.1, 0.1]
        p = [1, 1, eps, 0.5, 0.0]
        fhn=model.FHN(p)
        u0, v0 = np.meshgrid(np.linspace(-1.25,1.75,n_intiaL_conditions),np.linspace(-0.75,1.75,n_intiaL_conditions))
        x0 = np.array([u0,v0])
        fhn.generate_data(x0, dt, N)
    pass

def Bicubic(dt=0.1, N=1000000, n_intiaL_conditions=1):
    """
    Generate data using the Bicubic model (see src/pyCLINE/model.py),
    as used in Prokop, Billen, Frolov, Gelens (2025).

    Args:
        dt (float, optional): Time step. Defaults to 0.1.
        N (int, optional): Number of time steps. Defaults to 1000000.
    """
    u = np.zeros([2,N])

    u[:,0] = [0.1, 0.1]
    p = [-0.5, 0.5, -1/3]

    bicubic=model.Bicubic(p)
    u0, v0 = np.meshgrid(np.linspace(-1.25,1.75,n_intiaL_conditions),np.linspace(-0.75,1.75,n_intiaL_conditions))
    x0 = np.array([u0,v0])
    bicubic.generate_data(x0, dt, 10000)
    pass

def GeneExpression(dt=0.1, N=1000000, n_intiaL_conditions=1):
    """
    Generate data using the Gene Expression model (see src/pyCLINE/model.py),
    as used in Prokop, Billen, Frolov, Gelens (2025).

    Args:
        dt (float, optional): Time step. Defaults to 0.1.
        N (int, optional): Number of time steps. Defaults to 1000000.
    """
    u = np.zeros([2,N])

    u[:,0] = [0.1, 0.1]
    p=[1, 0.05,  1, 0.05,   1, 0.05,  1,  1, 0.1,  2]

    gene_expression=model.GeneExpression(p)
    u0, v0 = np.meshgrid(np.linspace(0,1.75,n_intiaL_conditions),np.linspace(0,1.75,n_intiaL_conditions))
    x0 = np.array([u0,v0])
    gene_expression.generate_data(x0, dt, 10000)
    pass

def DelayOscillator(N=20000):
    """
    Generate data using the Delay Oscillator model (see src/pyCLINE/model.py),
    as used in Prokop, Billen, Frolov, Gelens (2025).

    Args:
        N (int, optional): Number of time steps. Defaults to 20000.
    """
    time = np.linspace(0, 400, N-1)
    dt=time[1]-time[0]
    p=[4, 10, 2]

    delay_osci=model.DelayOscillator(p)
    delay_osci.generate_data( y_0=0, dt=dt, t_max=time[-1])
    pass