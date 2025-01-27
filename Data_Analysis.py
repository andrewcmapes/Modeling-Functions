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




class Data_Analysis:
    @staticmethod # Utilizes the standardized RMSE
    def RMSE(Data1, Data2):
        Sum = 0
        for i in range(0,len(Data1)):
            Sum += (Data1[i] - Data2[i])**2/len(Data1)
            
        return np.sqrt(Sum)/np.var(Data2)**(1/2)
    
    @staticmethod
    def Corr(Data1, Data2):
        Sum_1, Sum_2, Sum_3 = 0, 0, 0
        for i in range(0, len(Data1)):
            Sum_1 += (Data1[i]-np.mean(Data1))*(Data2[i]-np.mean(Data2))
            Sum_2 += (Data1[i]-np.mean(Data1))**2
            Sum_3 += (Data2[i]-np.mean(Data2))**2
        return Sum_1 / (np.sqrt(Sum_2) * np.sqrt(Sum_3))
    
    @staticmethod # Need to rethink how to handle natural data sets with this function. Currently dt breaks based off of dates being used for time in series data
    def Poly_OLS_SDE(Data, Terms = 2):
        dt = (Data[1,1]-Data[0,1])
        n = len(Data[:,0])
        x = Data[:,0] # Original Data
        dx = np.zeros((n, 1)) # Differntial series
        mod = np.zeros((n, Terms)) # Model Data
                
        for i in range(0,n-1):
            dx[i+1] = (x[i+1] - x[i])/dt
            for j in range(0, Terms):
                mod[i+1,j] = x[i]**j
        
        return np.linalg.inv(mod.T @ mod) @ np.dot(mod.T, dx)
    
    @staticmethod
    def Poly_OLS(Data, Terms = 2):
        n = len(Data[:,0])
        X_mod = np.zeros((n, Terms))
        
        for i in range(0,n):
            for j in range(0, Terms):
                X_mod[i,j] = Data[i,0]**j
        
        return np.linalg.inv(X_mod.T @ X_mod) @ np.dot(X_mod.T, Data[:,1])
        
    @staticmethod # Possibly complete, but accuracy is pretty variable depending on inputs
    def Noise_Modeler(Data, Beta, Bins=1000, Terms=1):
        dt = (Data[1,1]-Data[0,1])
        n = len(Data[:,0])
        X_noi = np.np.zeros((n, 1))
        Noise = np.zeros((Bins, 2))
        
        for i in range(1, n):
            Est = 0
            for j in range(0, len(Beta)):
                Est += Beta[j] * Data[i-1,0]**j * dt
            X_noi[i] = Data[i,0] - Data[i-1,0] - Est
        
        Per = np.linspace(5, 95, (Bins+1))
        Bin_edges = np.percentile(Data[:,0], Per)
        Masks = {}
        for i in range(0, Bins):
            Mask = (Data[:,0] >= Bin_edges[i]) & (Data[:,0] <= Bin_edges[i+1])
            Masks[f'bin_{i}'] = [Data[Mask,0], X_noi[Mask]]
        
        for i in range(0, Bins):
            Noise[i,0] = Masks[f'bin_{i}'][0].mean()
            Noise[i,1] = (Masks[f'bin_{i}'][1].var()/dt)**(1/2)
        Beta = Data_Analysis.Poly_OLS(Noise, Terms=Terms)
        return Beta
    
    @staticmethod
    def Ensemble_Distribution(Ensemble):
        n = len(Ensemble[:, 0])
        mean = np.zeros((n, 1))
        var = np.zeros((n, 1))
        for i in range(0, n):
            mean[i] = np.mean(Ensemble[i,0:-1])
            var[i] = np.var(Ensemble[i,0:-1])
        return mean, var
    
    @staticmethod
    def Ensemble_Forecast(Data, Lead, N=20, deterministic = 'y', noise = 'y', Bins = 1000, plot='on'):
        if deterministic == 'y':
            Beta = Data_Analysis.Poly_OLS_SDE(Data, Terms=2)
            deterministic = f'{Beta[1]}*x+{Beta[0]}'
            print(Beta)
        if noise == 'y':
            Sigma = Data_Analysis.Noise_Modeler(Data, Beta, Bins)
            noise = f'{Sigma[1]}*x+{Sigma[0]}'
            
        Time = int(np.round(Data[-1,1] - Data[0,1]))
        n = Time+2-Lead
        Points = np.zeros((n-1,2))
        for i in range(1, n):
            Mask = np.round(Data[:,1], 5) == i-1
            mean, var = Data_Analysis().Ensemble_Distribution(Data_Generator.Ensemble(Type='EM', N=N, deterministic = deterministic, noise = noise, T=Lead, X_0=float(Data[Mask, 0])))
            Points[i-1, 1] = mean[-1]
            if plot == 'on':
                plt.plot(np.array([i-1,i-1+Lead]), np.array([float(Data[Mask, 0]), Points[i-1,1]]), color='red', label="Ensemble Forecast" if i == 1 else "")
                plt.scatter((i-1+Lead), Points[i-1, 1], color='black')
        
        
        for i in range(Lead, Time+1):
            Mask = np.round(Data[:,1], 5) == i
            Points[i-Lead, 0] = float(Data[Mask, 0])
            
        if plot == 'on': 
            plt.plot(Data[:,1], Data[:,0], color = 'blue', label='Truth')
            plt.legend()
            plt.title(f'{Lead} Second Lead Time Ensemble Forecasts')
            plt.show()
        return Data_Analysis.Corr(Points[:,0], Points[:,1]), Data_Analysis.RMSE(Points[:,0], Points[:,1])
    