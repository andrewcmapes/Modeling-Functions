import matplotlib.pyplot as plt
import pandas_datareader.data as pdr
import numpy as np
import pandas as pd
import json # JSON is the JavaScript Object Notation library and is used in this code for converting python formats when working with the BLS API
import requests # Requests allows for easy interactions with HTTP web servers/APIs
from datetime import datetime
import sys
import inspect
import os
from io import StringIO



def Forward_Euler(drift = 'y', dt = 0.01, T = 50, X_0 = 0):
    """
    Forward Euler function is used to simulate data points of a deterministic differential equation.
    It works by requesting the string of your function. If the parameter entry is not used.
    Then using the eval function and a lambda expression string, the function is created and run
    through the forward Euler equations to generate the desired data

    The SDE is defined as:
        dX_t = drift(X_t, t) * dt
    where:
        - `drift` is the deterministic part of the DE.

    PARAMETERS:
        drift. TYPE: str
            The entry here should be a string that gives the equation of the function
            that is to be simulated
        dt. TYPE: float
            The entry here is for the time step
        T. TYPE: int
            The time duration of the simulated data
        X_0. TYPE: float
            The initial value of the simulated data    
    """

    if drift == 'y':
        drift = input("Enter your differential equation (e.g., 'x**2 + 2'): ")
    drift = eval(f"lambda x: {drift}")
    
    steps = int(np.round(T/dt)) + 1
    X = np.zeros((steps, 3))
    X[0, 0] = X_0
    
    for i in range(0,steps-1):
        X[i+1,0] = X[i,0] + drift(X[i,0]) * dt
        X[i+1,1] = X[i,1] + dt
    return X



def Euler_Maruyama(drift = 'y', diffusion = 'y', dt = 0.01, T = 50, X_0 = 0):
    """
    Simulates sample paths of a stochastic differential equation (SDE) using the Euler-Maruyama method.

    The SDE is defined as:
        dX_t = drift(X_t, t) * dt + diffusion(X_t, t) * dW_t
    where:
        - `drift` is the deterministic part of the SDE.
        - `diffusion` is the stochastic part of the SDE.
        - `W_t` is a Wiener process (Brownian motion).

    PARAMETERS:
        drift. TYPE: str
            drift is the portion of your SDE that has no influence from stochastic factors. The 
            parameter here can be used to avoid providing this string from the prompts. If not used, the 
            user will be prompted for the string when the function is called.
        diffusion. TYPE: str
            diffusion is the portion of your SDE that is influenced by a Weiner Process. Similar to the 
            drift parameter, the diffusion parameter can be used to avoid providing the string 
            from the input prompts after the function is run.
        dt. TYPE: float
            The entry here is for the time step
        T. TYPE: int
            The time duration of the simulated data
        X_0. TYPE: float
            The initial value of the simulated data
    """

    if drift == 'y':
        drift = input("Enter the drift portion of your SDE (e.g., 'x**2 + 2'): ")
    if "t" in drift:
        drift = eval(f"lambda x, t: {drift}")
    else:
        drift = eval(f"lambda x: {drift}")
        
    if diffusion == 'y':
        diffusion = input("Enter the stochastic portion of your SDE (e.g., 'x**2 + 2'): ")
    if "t" in diffusion:
        diffusion = eval(f"lambda x, t: {diffusion}")
    else:
        diffusion = eval(f"lambda x: {diffusion}")
    
    steps = int(np.round(T/dt)) + 1
    X = np.zeros((steps, 3))
    X[0, 0] = X_0
    
    for i in range(0,steps-1):
        X[i+1,1] = X[i,1] + dt
        X[i+1,2] = np.sqrt(dt) * np.random.normal(0,1)
        if "t" in inspect.signature(drift).parameters:
            det = drift(X[i,0], X[i,1])
        else:
            det = drift(X[i,0])
        if "t" in inspect.signature(diffusion).parameters:
            sto = diffusion(X[i,0], X[i,1])
        else:
            sto = diffusion(X[i,0])
        X[i+1,0] = X[i,0] + det * dt + sto * X[i,2]
        
    return X



def Milstein(drift = 'y', diffusion = 'y', diff_diffusion = 'y', dt = 0.01, T = 50, X_0 = 0):
    """
    Simulates sample paths of a stochastic differential equation (SDE) using the Milstein method.

    The SDE is defined as:
        dX_t = drift(X_t, t) * dt + diffusion(X_t, t) * dW_t
    where:
        - `drift` is the deterministic part of the SDE.
        - `diffusion` is the stochastic part of the SDE.
        - `W_t` is a Wiener process (Brownian motion).

    PARAMETERS:
        drift. TYPE: str
            Drift is the portion of your SDE that has no influence from stochastic factors. The 
            parameter here can be used to avoid providing this string from the prompts. If not used, the 
            user will be prompted for the string when the function is called.
        diffusion. TYPE: str
            Diffusion is the portion of your SDE that is influenced by a Weiner Process. Similar to the 
            drift parameter, the diffusion parameter can be used to avoid providing the string 
            from the input prompts after the function is run.
        diff_diffusion. TYPE: str
            The diff_diffusion parameter is looking for the derivative of the diffusion term in your 
            SDE. If not given to the parameter, the user will be prompted for this string when the function
            is run. 
        dt. TYPE: float
            The entry here is for the time step
        T. TYPE: int
            The time duration of the simulated data
        X_0. TYPE: float
            The initial value of the simulated data
    """
    if drift == 'y':
        drift = input("Enter the drift portion of your SDE (e.g., 'x**2 + 2'): ")
    drift = eval(f"lambda x: {drift}")
    if diffusion == 'y':
        diffusion = input("Enter the stochastic portion of your SDE (e.g., 'x**2 + 2'): ")
    diffusion = eval(f"lambda x: {diffusion}")
    if diff_diffusion == 'y':
        diff_diffusion = input("Enter the stochastic portion of your SDE's derivative (e.g., 'x**2 + 2'): ")
    diff_diffusion = eval(f"lambda x: {diff_diffusion}")
    
    steps = int(np.round(T/dt)) + 1
    X = np.zeros((steps, 3))
    X[0, 0] = X_0
    
    for i in range(0,steps-1):
        X[i+1,2] = np.sqrt(dt) * np.random.normal(0,1)
        X[i+1,0] = X[i,0] + drift(X[i,0]) * dt + diffusion(X[i,0]) * X[i,2] + (1/2) * diffusion(X[i,0]) * diff_diffusion(X[i,0]) * (X[i,2]**2-dt**2)
        X[i+1,1] = X[i,1] + dt
    return X



def Ensemble(method = 'EM', N = 10, drift = 'y', diffusion = 'y', dt = 0.01, T = 50, X_0 = 0):
    """
    Ensemble function is used to create a set of ensemble members for analysis from an (SDE) when a singular
    sample path is not sufficient. It is capable of either using Euler-Maruyama method or Forward Euler 
    depending on if there exists a stochastic portion of the SDE.

    The SDE is defined as:
        dX_t = drift(X_t, t) * dt + diffusion(X_t, t) * dW_t
    where:
        - `drift` is the deterministic part of the SDE.
        - `diffusion` is the stochastic part of the SDE.
        - `W_t` is a Wiener process (Brownian motion).

    PARAMETERS:
        method. TYPE: str
            method may either be 'EM' for Euler-Maruyama method of data generation for an SDE, or 'FE' for
            Forward Euler method of data generation for an ODE.
        N. TYPE: int
            N is the number of ensemble members desired. 
        drift. TYPE: str
            Drift is the portion of your SDE that has no influence from stochastic factors. The 
            parameter here can be used to avoid providing this string from the prompts. If not used, the 
            user will be prompted for the string when the function is called.
        diffusion. TYPE: str
            Diffusion is the portion of your SDE that is influenced by a Weiner Process. Similar to the 
            drift parameter, the diffusion parameter can be used to avoid providing the string 
            from the input prompts after the function is run.
        dt. TYPE: float
            The entry here is for the time step
        T. TYPE: int
            The time duration of the simulated data
        X_0. TYPE: float
            The initial value of the simulated data
    """
    n = int(T/dt) + 1
    if method == 'FE':
        if drift == 'y':
            drift = input("Enter the drift portion of your SDE (e.g., 'x**2 + 2'): ")
        Data = np.tile([0.0]*(N+1), (n,1))
        for i in range(0,N):
            X = Data_Generator.Forward_Euler(drift = drift, dt = dt, T = T, X_0 = X_0)
            Data[:,i] = X[:,0]
        Data[:,N] = X[:,1]
    
    elif method == 'EM':
        if drift == 'y':
            drift = input("Enter the drift portion of your SDE (e.g., 'x**2 + 2'): ")
        if diffusion == 'y':
            diffusion = input("Enter the stochastic portion of your SDE (e.g., 'x**2 + 2'): ")
        Data = np.tile([0.0]*(N+1), (n,1))
        for i in range(0,N):
            X = Data_Generator.Euler_Maruyama(drift = drift, diffusion = diffusion, dt = dt, T = T, X_0 = X_0)
            Data[:,i] = X[:,0]
        Data[:,N] = X[:,1]
        
    return Data



def Ordinary_Equation(equation = 'y', diffusion = 'y', dt = 0.01, T = 50, X_0 = 0):
    if equation == 'y':
        equation = input("Enter the Ordinary Equation (e.g., 'x**2 + 2'): ")
    else:
        equation = eval(f"lambda x: {equation}")
        
    if diffusion == 'y':
        diffusion = input("Enter the diffusion Equation (e.g., 'x**2 + 2'): ")
    else:
        diffusion = eval(f"lambda x: {diffusion}")
    
    steps = int(np.round(T/dt)) + 1
    X = np.zeros((steps, 3))
    X[0, 0] = X_0
    Time = 0
    for i in range(0,steps-1):
        Time += dt
        X[i,1] = Time
        X[i,2] = np.sqrt(dt) * np.random.normal(0,1)
        X[i,0] = equation(Time) + diffusion(equation(Time)) * X[i,2]
    return X



def Scatter_Points(n: int=100, grid: list=[0,10], dim=2) -> list:
    """
    Scatter Points is a function that will generate a list of arrays that have random values.

    PARAMETERS:
        n. TYPE: int
            n is the number of arrays desired
        grid. TYPE: list
            grid is the list of boundaries for the random values the dimensions of the array take on
        dim. TYPE: int
            dim is the dimension of the arrays desired
    """
    points = []
    for i in range(n):
        points.append(np.random.randint(low=grid[0], high=grid[-1], size=dim))
    return points
