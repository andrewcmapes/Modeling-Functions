from data_generator import ensemble
import matplotlib.pyplot as plt
import numpy as np
import os

def rmse(data1, data2):
    """
    The function calculates the normalized root mean squared error for 2 given data sets.

    Args:
        data1 (dataFrame): The first data set to be used
        data2 (dataFrame): The second data set to be used

    Returns:
        float: The value of the normalized root mean squared error
    """
    sum = 0
    for i in range(0, len(data1)):
        sum += (data1[i] - data2[i])**2/len(data1)
        
    return np.sqrt(sum)/np.var(data2)**(1/2)



def corr(data1, data2):
    """
    corr is a function that returns 2 data sets correlation

    Args:
        data1 (dataFrame): The first data set to be used
        data2 (dataFrame): The second data set to be used

    Returns:
        float: The value of the correlation between the two sets
    """
    sum_1, sum_2, sum_3 = 0, 0, 0
    for i in range(0, len(data1)):
        sum_1 += (data1[i]-np.mean(data1))*(data2[i]-np.mean(data2))
        sum_2 += (data1[i]-np.mean(data1))**2
        sum_3 += (data2[i]-np.mean(data2))**2
    return sum_1 / (np.sqrt(sum_2) * np.sqrt(sum_3))



def poly_ols_sde(data, terms = 2):
    """
    poly_ols_SDE is a function that uses a polynomial SDE form to model data.
    The functional form of the model is defined as
        dX_t = (a*x**2+b*x+c) * dt
    where:
        The highest order term is determined by the terms parameter

    Args:
        data (dataFrame): The time series data to be used for this differential equation model
        terms (int, optional): The degree of the leading term in the polynomial. Defaults to 2.

    Returns:
        list: Output is the coefficients for the model such as a,b,c from the example form above.
    """

    dt = (data[1,1]-data[0,1])
    n = len(data[:,0])
    x = data[:,0] # Original data
    dx = np.zeros((n, 1)) # Differntial series
    mod = np.zeros((n, terms)) # Model data
            
    for i in range(0,n-1):
        dx[i+1] = (x[i+1] - x[i])/dt
        for j in range(0, terms):
            mod[i+1,j] = x[i]**j
    
    return np.linalg.inv(mod.T @ mod) @ np.dot(mod.T, dx)



def poly_ols(data, terms = 2):
    """
    poly_ols_SDE is a function that uses a polynomial SDE form to model data.
    The functional form of the model is defined as
        dX_t = (a*x**2+b*x+c) * dt
    where:
        The highest order term is determined by the terms parameter

    Args:
        data (dataFrame): The time series data to be used for this differential equation model
        terms (int, optional): The degree of the leading term in the polynomial. Defaults to 2.

    Returns:
        list: Output is the coefficients for the model such as a,b,c from the example form above.
    """
    n = len(data[:,0])
    x_mod = np.zeros((n, terms))
    
    for i in range(0,n):
        for j in range(0, terms):
            x_mod[i,j] = data[i,0]**j
    
    return np.linalg.inv(x_mod.T @ x_mod) @ np.dot(x_mod.T, data[:,1])



def noise_modeler(data, beta, bins=1000, terms=1):
    dt = (data[1,1]-data[0,1])
    n = len(data[:,0])
    x_noi = np.np.zeros((n, 1))
    noise = np.zeros((bins, 2))
    
    for i in range(1, n):
        Est = 0
        for j in range(0, len(beta)):
            Est += beta[j] * data[i-1,0]**j * dt
        x_noi[i] = data[i,0] - data[i-1,0] - Est
    
    per = np.linspace(5, 95, (bins+1))
    bin_edges = np.percentile(data[:,0], per)
    masks = {}
    for i in range(0, bins):
        mask = (data[:,0] >= bin_edges[i]) & (data[:,0] <= bin_edges[i+1])
        masks[f'bin_{i}'] = [data[mask,0], x_noi[mask]]
    
    for i in range(0, bins):
        noise[i,0] = masks[f'bin_{i}'][0].mean()
        noise[i,1] = (masks[f'bin_{i}'][1].var()/dt)**(1/2)
    beta = poly_ols(noise, terms=terms)
    return beta



def ensemble_distribution(ensemble):
    n = len(ensemble[:, 0])
    mean = np.zeros((n, 1))
    var = np.zeros((n, 1))
    for i in range(0, n):
        mean[i] = np.mean(ensemble[i,0:-1])
        var[i] = np.var(ensemble[i,0:-1])
    return mean, var



def ensemble_forecast(data, lead, N=20, drift = 'y', noise = 'y', bins = 1000, plot='on'):
    if drift == 'y':
        beta = poly_ols_sde(data, terms=2)
        drift = f'{beta[1]}*x+{beta[0]}'
        print(beta)
    if noise == 'y':
        sigma = noise_modeler(data, beta, bins)
        noise = f'{sigma[1]}*x+{sigma[0]}'
        
    time = int(np.round(data[-1,1] - data[0,1]))
    n = time+2-lead
    points = np.zeros((n-1,2))
    for i in range(1, n):
        mask = np.round(data[:,1], 5) == i-1
        mean, var = ensemble_distribution(ensemble(method='em', N=N, drift = drift, noise = noise, t=lead, x_0=float(data[mask, 0])))
        points[i-1, 1] = mean[-1]
        if plot == 'on':
            plt.plot(np.array([i-1,i-1+lead]), np.array([float(data[mask, 0]), points[i-1,1]]), color='red', label="ensemble forecast" if i == 1 else "")
            plt.scatter((i-1+lead), points[i-1, 1], color='black')
    
    
    for i in range(lead, time+1):
        mask = np.round(data[:,1], 5) == i
        points[i-lead, 0] = float(data[mask, 0])
        
    if plot == 'on': 
        plt.plot(data[:,1], data[:,0], color = 'blue', label='Truth')
        plt.legend()
        plt.title(f'{lead} Second lead time ensemble forecasts')
        plt.show()
    return corr(points[:,0], points[:,1]), rmse(points[:,0], points[:,1])
