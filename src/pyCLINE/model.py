import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import jitcdde as jitcdde

# -------------------  FitzHugh-Nagumo oscillator -------------------
class FHN:
    """
    The Fitzhuge-Nagumo model is a simplified model of the electrical activity of a neuron. In this form taken from Prokop et al., iScience (2024).
    """    
    def __init__(self,  p=[1, 1, 0.3, 0.5, 0.0]):
        self.p = p # parameters of the model

    def model(self, U): 
        """
        Model formulation of the FHN model.
        With: 
        p = [c, d, eps, b, a]
        p = [1, 1, 0.3, 0.5, 0.0]
        Prokop et al., iScience (2024)

        Args:
            U (array): Array of initial conditions for the model.

        Returns:
            U (array): Array of the next state of the model.
        """    
        u= U[0]
        v= U[1]
        # p = [c, d, eps, b, a];   Prokop et al., iScience (2024)
        # p = [1, 1, 0.3, 0.5, 0.0]
        return np.array([-u**3 + self.p[0]*u**2 + self.p[1]*u - v,
                        self.p[2]*(u - self.p[3]*v + self.p[4])])

    def vnull(self, u):
        """
        Calculate the v-nullcline of the FHN model

        Args:
            u (array): Array of values of the v-variable.

        Returns:
            v: Array of values of the v-nullcline. 
        """
        return -u**3 + self.p[0]*u**2 + self.p[1]*u

    def unull(self, v):
        """
        Calculate the u-nullcline of the FHN model

        Args:
            v (array): Array of values of the v-variable.

        Returns:
            u: Array of values of the u-nullcline. 
        """
        return  self.p[3]*v - self.p[4]

    def fixed_points(self):
        """
        Calculate the fixed points of the FHN model.

        Returns:
            fp (Array): Array of fixed points of the model.
        """
        sol = np.roots([-self.p[3], self.p[3]*self.p[0], (self.p[1]*self.p[3]-1), -self.p[4]])

        rids = np.where(np.imag(sol)==0)
        rsol = sol[rids]

        fp = np.zeros([2,rsol.shape[0]])

        fp[0,:] = np.real(rsol)
        fp[1,:] = (fp[0,:] + self.p[4])/self.p[3]

        return fp

    def simulate(self, U, dt):
        """
        Generates the next state of the FHN model using the Runge-Kutta 4th order solver.

        Args:
            U (array): Array of initial conditions for the model.
            dt (float): Time step.

        Returns:
            U (array): Array of the next state of the model.
        """
        return rk4_solver(self.model, U, dt)
    
    def generate_data(self, x0, dt, N=10000, save=True, plot=False, max_time=2, check_period=False):
        """
        Generates synthetic data for the FHN model.

        Args:
            x0 (array): Array of initial conditions for the model.
            dt (float): Time step.
            N (int, optional): Number of time steps. Defaults to 10000.
            save (bool, optional): Save the generated data to a csv file. Defaults to True.
            plot (bool, optional): Plot the generated data. Defaults to False.
            max_time (int, optional): Maximum time for the plot. Defaults to 2.
            check_period (bool, optional): Check the period of the oscillation. Defaults to False.
        """
        df=simulate_data(self.simulate, x0, dt, N, check_period)
        if save:
            save_data(df, f'{self.__class__.__name__}_eps={self.p[2]}_a={self.p[4]}.csv')
        if plot:
            plot_data(df, max_time=max_time)


# -------------------  bicubic oscillator -------------------
class Bicubic:
    """
    The bicubic model if a model with two nonlinear nullclines (one bistable and one s-shaped) taken from Prokop et al., Chaos (2024).
    """    
    def __init__(self,p=[-0.5, 0.5, -0.3]):
        self.p = p

    def model(self, U):
        """
        Model formulation of the bicubic model.
        With: 
        p = [-0.5, 0.5, -0.3];   
        Prokop et al., Chaos (2024)

        Args:
            U (array): Array of initial conditions for the model.

        Returns:
            U (array): Array of the next state of the model.
        """  
        
        u= U[0]
        v= U[1]
        return np.array([-u**3 + u**2 + u - v,
                        self.p[0]*v**3 + self.p[1]*v**2 + self.p[2]*v + u])

    def vnull(self, u):
        """
        Calculate the v-nullcline of the bicubic model

        Args:
            u (array): Array of values of the v-variable.

        Returns:
            v: Array of values of the v-nullcline. 
        """
        return -u**3 + u**2 + u

    def unull(self, v):
        """
        Calculate the u-nullcline of the bicubic model

        Args:
            v (array): Array of values of the v-variable.

        Returns:
            u: Array of values of the u-nullcline. 
        """
        return  -(self.p[0]*v**3 + self.p[1]*v**2 + self.p[2]*v)
    
    def simulate(self, U, dt):
        """
        Generates the next state of the bicubic model using the Runge-Kutta 4th order solver.

        Args:
            U (array): Array of initial conditions for the model.
            dt (float): Time step.

        Returns:
            U (array): Array of the next state of the model.
        """
        return rk4_solver(self.model, U, dt)
    
    def generate_data(self, x0, dt, N=10000, save=True, plot=False, max_time=2, check_period=False):
        """
        Generates synthetic data for the biubic model.

        Args:
            x0 (array): Array of initial conditions for the model.
            dt (float): Time step.
            N (int, optional): Number of time steps. Defaults to 10000.
            save (bool, optional): Save the generated data to a csv file. Defaults to True.
            plot (bool, optional): Plot the generated data. Defaults to False.
            max_time (int, optional): Maximum time for the plot. Defaults to 2.
            check_period (bool, optional): Check the period of the oscillation. Defaults to False.
        """
        df=simulate_data(self.simulate, x0, dt, N, check_period)
        if save:
            save_data(df, f'{self.__class__.__name__}.csv')
        if plot:
            plot_data(df, max_time=max_time)

# -------------------  gene expression oscillator -------------------
class GeneExpression:
    """
    The gene expression model is a model of the regulation of gene expression taken from Novak & Tyson, Nature Rev Mol Cell Biol (2008).
    """    
    def __init__(self, p=[1, 0.05,  1, 0.05,   1, 0.05,  1,  1, 0.1,  2]):
        self.p = p

    def model(self, U):
        """
        Model formulation of the gene expression model.
        With: 
        p = [S,   k1, Kd,  kdx, ksy,  kdy, k2, ET,  Km, KI]
        p = [1, 0.05,  1, 0.05,   1, 0.05,  1,  1, 0.1,  2];   
        Novak & Tyson, Nature Rev Mol Cell Biol (2008)

        Args:
            U (array): Array of initial conditions for the model.

        Returns:
            U (array): Array of the next state of the model.
        """
        u= U[0]
        v= U[1]
        # p = [S,   k1, Kd,  kdx, ksy,  kdy, k2, ET,  Km, KI]
        # p = [1, 0.05,  1, 0.05,   1, 0.05,  1,  1, 0.1,  2];   Novak & Tyson, Nature Rev Mol Cell Biol (2008)
        return np.array([self.p[4]*v - self.p[5]*u - self.p[6]*self.p[7]*u/(self.p[8] + u + self.p[9]*u**2),
                        self.p[1]*self.p[0]*self.p[2]**4/(self.p[2]**4 + u**4) - self.p[3]*v])

    def vnull(self, v): 
        """
        Calculate the v-nullcline of the gene expression model

        Args:
            v (array): Array of values of the v-variable.

        Returns:
            u: Array of values of the v-nullcline. 
        """
        return self.p[1]*self.p[0]/self.p[3]*self.p[2]**4/(self.p[2]**4 + v**4)

    def unull(self, v):
        """
        Calculate the u-nullcline of the gene expression model

        Args:
            v (array): Array of values of the v-variable.

        Returns:
            u: Array of values of the u-nullcline. 
        """
        return self.p[5]*v/self.p[4] + self.p[6]*self.p[7]*v/(self.p[8] + v + self.p[9]*v**2)

    def simulate(self, U, dt):
        """
        Generates the next state of the gene expression model using the Runge-Kutta 4th order solver.

        Args:
            U (array): Array of initial conditions for the model.
            dt (float): Time step.

        Returns:
            U (array): Array of the next state of the model.
        """
        return rk4_solver(self.model, U, dt)
    
    def generate_data(self, x0, dt, N=10000, save=True, plot=False, max_time=10, check_period=False):
        """
        Generates synthetic data for the gene expression model.

        Args:
            x0 (array): Array of initial conditions for the model.
            dt (float): Time step.
            N (int, optional): Number of time steps. Defaults to 10000.
            save (bool, optional): Save the generated data to a csv file. Defaults to True.
            plot (bool, optional): Plot the generated data. Defaults to False.
            max_time (int, optional): Maximum time for the plot. Defaults to 10.
            check_period (bool, optional): Check the period of the oscillation. Defaults to False.
        """
        df=simulate_data(self.simulate, x0, dt, N, check_period)
        if save:
            save_data(df, f'{self.__class__.__name__}.csv')
        if plot:
            plot_data(df, max_time=max_time)

# -------------------  Gylcolytic oscillator -------------------
class GlycolyticOscillations:
    """
    The glycolytic oscillator is a model of glycolytic oscillations introduced in Prokop et al., iScience (2024) and in 
    this form taken from Prokop et al., Chaos (2024).
    """
    def __init__(self, p=[-0.3, -2.2, 0.25, -0.5, 0.5, 1.8, 0.7, -0.3]):
        self.p = p

    def model(self, U):
        """
        Model formulation of the glycolytic oscillations model.
        With: 
        p = [a, b, c, d, e, f, g, h]
        p = [-0.3, -2.2, 0.25, -0.5, 0.5, 1.8, 0.7, -0.3]  
        Prokop et al., Chaos (2024)

        Args:
            U (array): Array of initial conditions for the model.

        Returns:
            U (array): Array of the next state of the model.
        """
        u= U[0]
        v= U[1]
        
        return np.array([self.p[0]*u + self.p[1]*v + self.p[2]*u**2 + self.p[3]*u**3 + self.p[4]*v**3,
                        self.p[5]*u+self.p[6]*v+self.p[7]*u**3])

    def simulate(self, U, dt):
        """
        Generates the next state of the glycolytic oscillations model using the Runge-Kutta 4th order solver.

        Args:
            U (array): Array of initial conditions for the model.
            dt (float): Time step.

        Returns:
            U (array): Array of the next state of the model.
        """
        return rk4_solver(self.model, U, dt)
    
    def generate_data(self, x0, dt, N=10000, save=True, plot=False, max_time=10, check_period=False):
        """
        Generates synthetic data for the glycolytic oscillations model.

        Args:
            x0 (array): Array of initial conditions for the model.
            dt (float): Time step.
            N (int, optional): Number of time steps. Defaults to 10000.
            save (bool, optional): Save the generated data to a csv file. Defaults to True.
            plot (bool, optional): Plot the generated data. Defaults to False.
            max_time (int, optional): Maximum time for the plot. Defaults to 10.
            check_period (bool, optional): Check the period of the oscillation. Defaults to False.
        """
        df=simulate_data(self.simulate, x0, dt, N, check_period)
        if save:
            save_data(df, f'{self.__class__.__name__}.csv')
        if plot:
            plot_data(df, max_time=max_time)

# -------------------  Goodwin oscillator -------------------
class Goodwin:
    """
    The Goodwin model is a three dimensional model for phosphorylation/dephosphorylation processes of a transcription factor,
    taken in this form from Gonze et al., Acta Biotheoretica (2021), parameters from Prokop et al. (2024).
    """    
    def __init__(self, p=[1,1,1,0.1,0.1,0.1,10,1]):
        self.p = p

    def model(self, U):
        """
        Model formulation of the Goodwin model.
        With: 
        p = [a,b,c,d,e,f,n,K]
        p = [1,1,1,0.1,0.1,0.1,10,1]
        Prokop et al., iScience (2024)

        Args:
            U (array): Array of initial conditions for the model.

        Returns:
            U (array): Array of the next state of the model.
        """
        u= U[0]
        v= U[1]
        w= U[2]
        # p = [a,b,c,d,e,f,n,K]
        # p = [1,1,1,0.1,0.1,0.1,10,1];  from Prokop et al. (2024)
        return np.array([self.p[0]*(self.p[7]**self.p[6])/(self.p[7]**self.p[6] + w**self.p[6]) - self.p[3]*u,
                        self.p[1]*u - self.p[4]*v,
                        self.p[2]*v - self.p[5]*w])
    def unull(self, v, w):
        """
        Calculate the u-nullcline of the Goodwin model

        Args:
            v (array): Array of values of the v-variable.
            w (array): Array of values of the w-variable.

        Returns:
            u: Array of values of the u-nullcline. 
        """
        return self.p[0]/self.p[3]*((self.p[7]**self.p[6])/(self.p[7]**self.p[6] + w**self.p[6])) + 0 * v
    
    def vnull(self, u, w): 
        """
        Calculate the v-nullcline of the Goodwin model

        Args:
            u (array): Array of values of the v-variable.
            w (array): Array of values of the w-variable.

        Returns:
            v: Array of values of the v-nullcline. 
        """
        return self.p[1]*u/self.p[4] + 0 * w

    def wnull(self, u, v):
        """
        Calculate the w-nullcline of the Goodwin model

        Args:
            u (array): Array of values of the u-variable.
            v (array): Array of values of the v-variable.

        Returns:
            w: Array of values of the w-nullcline. 
        """
        return self.p[2]*v/self.p[5] + 0 * u

    def simulate(self, U, dt):
        """
        Generates the next state of the Goodwin model using the Runge-Kutta 4th order solver.

        Args:
            U (array): Array of initial conditions for the model.
            dt (float): Time step.

        Returns:
            U (array): Array of the next state of the model.
        """
        return rk4_solver(self.model, U, dt)
    
    def generate_data(self, x0, dt, N=10000, save=True, plot=False, max_time=10, check_period=False):
        """
        Generates synthetic data for the gene expression model.

        Args:
            x0 (array): Array of initial conditions for the model.
            dt (float): Time step.
            N (int, optional): Number of time steps. Defaults to 10000.
            save (bool, optional): Save the generated data to a csv file. Defaults to True.
            plot (bool, optional): Plot the generated data. Defaults to False.
            max_time (int, optional): Maximum time for the plot. Defaults to 10.
            check_period (bool, optional): Check the period of the oscillation. Defaults to False.
        """
        df=simulate_data(self.simulate, x0, dt, N, check_period)
        if save:
            save_data(df, f'{self.__class__.__name__}.csv')
        if plot:
            plot_data(df, max_time=max_time)

# -------------------  Oregonator oscillator -------------------

class Oregonator:
    """
    The Oregonator model is a model of the Belousov-Zhabotinsky reaction, taken from Tyson, Journal of Chemical Physics (1975).
    """
    def __init__(self, p = [0.005,3,0.60,1e-2]):
        self.p = p
    
    def model(self, U):
        """
        Model formulation of the Oregonator model.
        With: 
        p = [q,p,f,e]
        p = [0.005,3,0.60,1e-2]   
        Tyson, Journal of Chemical Physics (1975)

        Args:
            U (array): Array of initial conditions for the model.

        Returns:
            U (array): Array of the next state of the model.
        """
        x, y, z = U
        #1977 Tyson
        q=self.p[0]
        p=self.p[1]
        f=self.p[2] #0.1-2.0
        e=self.p[3]
        a=1-f+(3*f)*q/(1-f)
        b=(1-f)/(q)-(1-3*f)/(1-f)
        g=f-(f*q)/(1-f)
        d=(1-f)/(q)+(1+f)/(1-f)

        dx=(1/e)*(-a*x-b*y-q*x**2-x*y)
        dy=-g*x-d*y+f*z-x*y
        dz=(1/p)*(x-z)
        return np.array([dx, dy, dz])
      
    def xnull(self, x, z):
        """
        Calculate the x-nullcline of the Oregonator model

        Args:
            x (array): Array of values of the x-variable.
            z (array): Array of values of the z-variable.

        Returns:
            y: Array of values of the x-nullcline. 
        """
        q=self.p[0]
        p=self.p[1]
        f=self.p[2] #0.1-2.0
        e=self.p[3]
        a=1-f+(3*f)*q/(1-f)
        b=(1-f)/(q)-(1-3*f)/(1-f)
        g=f-(f*q)/(1-f)
        d=(1-f)/(q)+(1+f)/(1-f)
        return (-a*x-q*x**2)/(b+x) + 0*z
    
    def ynull(self, x, y):
        """
        Calculate the y-nullcline of the Oregonator model

        Args:
            x (array): Array of values of the x-variable.
            y (array): Array of values of the y-variable.

        Returns:
            z: Array of values of the y-nullcline. 
        """
        q=self.p[0]
        p=self.p[1]
        f=self.p[2] #0.1-2.0
        e=self.p[3]
        a=1-f+(3*f)*q/(1-f)
        b=(1-f)/(q)-(1-3*f)/(1-f)
        g=f-(f*q)/(1-f)
        d=(1-f)/(q)+(1+f)/(1-f)
        return (1/f)*(g*x+d*y+x*y)

    def znull(self, x, y):
        """
        Calculate the x-nullcline of the Oregonator model

        Args:
            x (array): Array of values of the x-variable.
            y (array): Array of values of the y-variable.

        Returns:
            z: Array of values of the z-nullcline. 
        """
        return x + 0*y
    
    def simulate(self, U, dt):
        """
        Generates the next state of the Oregonator model using the Runge-Kutta 4th order solver.

        Args:
            U (array): Array of initial conditions for the model.
            dt (float): Time step.

        Returns:
            U (array): Array of the next state of the model.
        """
        return rk4_solver(self.model, U, dt)

    def generate_data(self, x0, dt, N=10000, save=True, plot=False, max_time=10, check_period=False):
        """
        Generates synthetic data for the Oregonator model.

        Args:
            x0 (array): Array of initial conditions for the model.
            dt (float): Time step.
            N (int, optional): Number of time steps. Defaults to 10000.
            save (bool, optional): Save the generated data to a csv file. Defaults to True.
            plot (bool, optional): Plot the generated data. Defaults to False.
            max_time (int, optional): Maximum time for the plot. Defaults to 10.
            check_period (bool, optional): Check the period of the oscillation. Defaults to False.
        """
        df=simulate_data(self.simulate, x0, dt, N, check_period)
        if save:
            save_data(df, f'{self.__class__.__name__}.csv')
        if plot:
            plot_data(df, max_time=max_time)

class Lorenz:
    """
    The Lorenz model is a simple model of atmospheric convection, taken from Lorenz, Journal of the Atmospheric Sciences (1963).
    """
    def __init__(self, p = [-0.5,65]):
        self.p = p
    
    def model(self, U):
        """
        Model formulation of the Lorenz model.
        With: 
        p = [a, rho]
        p = [-0.5,65]

        Args:
            U (array): Array of initial conditions for the model.

        Returns:
            U (array): Array of the next state of the model.
        """
        x,y,z = U
        a, rho = self.p
        return np.array([a*rho*(x-y)-a*y*z, rho*x - y -x*z, -z +x*y])
    
    def xnull(self, y, z):
        """
        Calculate the x-nullcline of the Lorenz model

        Args:
            y (array): Array of values of the y-variable.
            z (array): Array of values of the z-variable.

        Returns:
            x: Array of values of the x-nullcline. 
        """
        a, rho = self.p
        return (1/rho)*y*z +y
    
    def ynull(self, x, z):
        """
        Calculate the y-nullcline of the Lorenz model

        Args:
            x (array): Array of values of the x-variable.
            z (array): Array of values of the z-variable.

        Returns:
            y: Array of values of the y-nullcline. 
        """
        a, rho = self.p
        return x*z - rho*x
    
    def znull(self, x, y):
        """
        Calculate the z-nullcline of the Lorenz model

        Args:
            x (array): Array of values of the x-variable.
            y (array): Array of values of the y-variable.

        Returns:
            z: Array of values of the z-nullcline. 
        """
        return x*y
    
    def simulate(self, U, dt):
        """
        Generates the next state of the Lorenz model using the Runge-Kutta 4th order solver.

        Args:
            U (array): Array of initial conditions for the model.
            dt (float): Time step.

        Returns:
            U (array): Array of the next state of the model.
        """
        return rk4_solver(self.model, U, dt)
    
    def generate_data(self, x0, dt, N=10000, save=True, plot=False, max_time=25, check_period=False):
        """
        Generates synthetic data for the Oregonator model.

        Args:
            x0 (array): Array of initial conditions for the model.
            dt (float): Time step.
            N (int, optional): Number of time steps. Defaults to 10000.
            save (bool, optional): Save the generated data to a csv file. Defaults to True.
            plot (bool, optional): Plot the generated data. Defaults to False.
            max_time (int, optional): Maximum time for the plot. Defaults to 25.
            check_period (bool, optional): Check the period of the oscillation. Defaults to False.
        """
        df=simulate_data(self.simulate, x0, dt, N, check_period)
        if save:
            save_data(df, f'{self.__class__.__name__}.csv')
        if plot:
            plot_data(df, max_time=max_time)

class Roessler:
    """
    The Roessler model is a simple model of a chaotic oscillator, taken from Roessler, Zeitschrift für Naturforschung A (1976).
    """
    def __init__(self, p = [0.2,0.2,1]):
        self.p = p
    
    def model(self, U):
        """
        Model formulation of the Roessler model.
        With: 
        p = [abc]
        p = [0.2,0.2,1]   
        Roessler, Zeitschrift für Naturforschung A (1976)

        Args:
            U (array): Array of initial conditions for the model.

        Returns:
            U (array): Array of the next state of the model.
        """
        x,y,z = U
        a,b,c=self.p
        dx=-y-z
        dy = x+a*y
        dz = b+(x-c)*z
        return np.array([dx, dy, dz])
    
    def xnull(self, x, z):
        """
        Calculate the x-nullcline of the Roessler model

        Args:
            x (array): Array of values of the x-variable.
            z (array): Array of values of the z-variable.

        Returns:
            y: Array of values of the x-nullcline. 
        """
        a,b,c=self.p
        return -z
    
    def ynull(self, x, z):
        """
        Calculate the y-nullcline of the Roessler model

        Args:
            x (array): Array of values of the x-variable.
            z (array): Array of values of the z-variable.

        Returns:
            y: Array of values of the y-nullcline. 
        """
        a,b,c=self.p
        return -(1/a)*x
    
    def znull(self, x, y):
        """
        Calculate the z-nullcline of the Roessler model

        Args:
            x (array): Array of values of the x-variable.
            y (array): Array of values of the y-variable.

        Returns:
            z: Array of values of the z-nullcline. 
        """
        a,b,c=self.p
        return b/(c-x)
    
    def simulate(self, U, dt):
        """
        Generates the next state of the Roessler model using the Runge-Kutta 4th order solver.

        Args:
            U (array): Array of initial conditions for the model.
            dt (float): Time step.

        Returns:
            U (array): Array of the next state of the model.
        """
        return rk4_solver(self.model, U, dt)
    
    def generate_data(self, x0, dt, N=10000, save=True, plot=False, max_time=25, check_period=False):
        """
        Generates synthetic data for the Roessler model.

        Args:
            x0 (array): Array of initial conditions for the model.
            dt (float): Time step.
            N (int, optional): Number of time steps. Defaults to 10000.
            save (bool, optional): Save the generated data to a csv file. Defaults to True.
            plot (bool, optional): Plot the generated data. Defaults to False.
            max_time (int, optional): Maximum time for the plot. Defaults to 25.
            check_period (bool, optional): Check the period of the oscillation. Defaults to False.
        """
        df=simulate_data(self.simulate, x0, dt, N, check_period)
        if save:
            save_data(df, f'{self.__class__.__name__}.csv')
        if plot:
            plot_data(df, max_time=max_time)

# -------------------  Delay oscillator -------------------
class DelayOscillator:
    """
    Simple model of a Delay Oscillator with a single delay, inspired by Lewis, Current Biology (2003)
    """    
    def __init__(self, p=[4,10,2]):
        self.p = p
    
    def model(self):
        """
        Model formulation of the delay oscillator model.
        With: 
        p = [beta, tau, n]
        p = [4,10,2]

        Args:
            U (array): Array of initial conditions for the model.

        Returns:
            U (array): Array of the next state of the model.
        """
        self.DDE = [self.p[0]/(1+jitcdde.y(0,jitcdde.t-self.p[1])**self.p[2])-jitcdde.y(0)]
        return self.DDE
    
    def unull(self, y):
        """
        Calculate the u-nullcline of the delay oscillator model

        Args:
            y (array): Array of values of the v-variable.

        Returns:
            x: Array of values of the u-nullcline. 
        """
        return self.p[0]/(1+y**self.p[2])

    def vnull(self, x):
        """
        Calculate the u-nullcline of the delay oscillator model

        Args:
            x (array): Array of values of the u-variable.

        Returns:
            y: Array of values of the v-nullcline. 
        """
        return (self.p[0]/x-1)**(1/self.p[2])
    
    def simulate(self,  dt, t_max, y_0=0):
        """
        Simulates the delay oscillatory model using the jitcdde solver.

        Args:
            dt (float): Time step.
            t_max (float): Maximum time of the simulation.
            y_0 (int, optional): Initial condition. Defaults to 0.

        Returns:
            array : Simulated data of the model.
        """
        self.model()
        DDE = jitcdde.jitcdde(self.DDE)
        DDE.constant_past(y_0)
        DDE.step_on_discontinuities()

        data = []
        for time in np.arange(DDE.t, DDE.t  +t_max, dt):
            data.append(DDE.integrate(time))
        data = np.array(data)

        return data

    def generate_data(self, y_0, dt, t_max, save=True, plot=False, check_period=False):
        """
        Generates synthetic data for the delay oscillator model.

        Args:
            y_0 (float): Initial condition for the model.
            dt (float): Time step.
            t_max (float): Maximum time of the simulation.
            save (bool, optional): Save the generated data to a csv file. Defaults to True.
            plot (bool, optional): Plot the generated data. Defaults to False.
            check_period (bool, optional): Check the period of the oscillation. Defaults to False.
        """
        data = self.simulate( dt, t_max,y_0)
        df = pd.DataFrame(data, columns=['u'])
        df['time'] = np.arange(data.shape[0])*dt
        if save:
            save_data(df, f'{self.__class__.__name__}.csv')
        if plot:
            plot_data(df, max_time=t_max)        
        
        

# -------------------  4th-order Runge-Kutta solver -------------------
    
def rk4_solver(f, u, dt):
    """
    Rung-Kutta 4th order solver for a system of ODEs.

    Args:
        f (function): Function that describes the system of ODEs. For model classes it is the model method.
        u (float): Value of a state variable at time t.
        dt (float): Time step.

    Returns:
        u[i+1] (float): Value of a state variable at time t+dt.
    """    
    k1 = f(u)
    k2 = f(u + k1*dt/2)
    k3 = f(u + k2*dt/2)
    k4 = f(u + k3*dt)

    return u + dt/6*(k1 + 2*k2 + 2*k3 + k4)

# -------------------  Generate and save data of multiple IC to csv -------------------

def simulate_data (simulate, x0, dt, N=10000, check_period=False):
    """
    Generate synthetic data for a given model and initial conditions.

    Args:
        simulate (method): Method that simulates the model. For model classes it is the simulate method.
        x0 (array of floats): Initial conditions for the model.
        dt (float): Time step.
        N (int, optional): Number of timesteps. Defaults to 10000.
        check_period (bool, optional): Limits the amount of periods calculated. Defaults to True.

    Returns:
        df (pandas DataFrame): DataFrame with the synthetic data.
    """    
    df = pd.DataFrame()
    sim_count = 1
    time = np.arange(N)*dt
    # print(x0,len(x0), N)
    u = np.zeros((len(x0),N))
    if len(x0)==2:
        for i in range(x0[0,:,:].shape[0]):
            for j in range(x0[0,:,:].shape[1]):
                
                u[:,0] = [x0[0,i,j], x0[1,i,j]]
                

                for n in range(1,N):
                    u[:,n] = simulate(u[:,n-1], dt)
                
                df_sim = pd.DataFrame()

                df_sim['sim'] = np.ones(u.shape[1])*sim_count
                df_sim['time'] = time
                var_name=['u', 'v', 'w', 'x', 'y']
                for i_var in range(len(x0)):
                    df_sim[f'{var_name[i_var]}'] = u[i_var,:]
                
                
        
                if check_period:
                    _,_,period,_=calculate_period(df_sim['u'].to_numpy(), df_sim['time'].to_numpy())
                    if period is not None:
                        max_time = 6 * period
                        df_sim = df_sim[df_sim['time'] <= max_time]

                sim_count += 1

                df = pd.concat((df, df_sim), ignore_index=True)
    if len(x0)==3:
        for i in range(x0[0,:,:,:].shape[0]):
            for j in range(x0[0,:,:,:].shape[1]):
                for k in range(x0[0,:,:,:].shape[2]):
                    u[:,0] = [x0[0,i,j,k], x0[1,i,j,k], x0[2,i,j,k]]

                    for n in range(1,N):
                        u[:,n] = simulate(u[:,n-1], dt)
                    
                    df_sim = pd.DataFrame()

                    df_sim['sim'] = np.ones(u.shape[1])*sim_count
                    df_sim['time'] = time
                    var_name=['u', 'v', 'w']
                    for i_var in range(len(x0)):
                        df_sim[f'{var_name[i_var]}'] = u[i_var,:]
                    
                    if check_period:
                        _,_,period,_=calculate_period(df_sim['u'].to_numpy(), df_sim['time'].to_numpy())
                        if period is not None:
                            max_time = 6 * period
                            df_sim = df_sim[df_sim['time'] <= max_time]

                    sim_count += 1

                    df = pd.concat((df, df_sim), ignore_index=True)

    return df

def save_data(df, filename):
    """
    Save the generated data to a csv file.

    Args:
        df (pandas DataFrame): DataFrame with the synthetic data.
        filename (str): Name of the file, usually taking the name of the model.
    """    
    # Create 'data' directory if it does not exist
    if not os.path.exists('data/synthetic_data'):
        os.makedirs('data/synthetic_data')

    # Save the DataFrame to the 'data' directory
    filepath = os.path.join('data/synthetic_data', filename)
    
    # Save the DataFrame to the 'data/synthetic_data' directory
    df.to_csv(filepath, index=False)
    print('Generated data saved to', filepath)


def plot_data(df, max_time):
    """
    Plot the generated data.

    Args:
        df (pandas DataFrame): DataFrame with the synthetic data.
        max_time (float): Maximum time for the plot.
    """    
    fig, ax = plt.subplots(1,1)

    for i in df['sim'].unique():
        df_sim = df[(df['sim']==i) & (df['time']<=max_time)].copy()
        df_sim.plot.line(x='u', y='v', ax=ax, legend=False)

    ax.set_ylabel('v')


from scipy.signal import find_peaks

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