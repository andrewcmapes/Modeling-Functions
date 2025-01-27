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



class Data_Visualization:
    @staticmethod
    def Ensemble_Plot(Ensemble, Truth = None, Plot_Ensemble = 'Ensemble', Plot_Truth = 'PDF-Evo', mean = 'y', var = 'y'):
        
        if Truth.any() != None:
            if Plot_Truth == 'PDF-Evo':
                Data_Visualization.Plot(Truth, Style = 'PDF-Evo', mean = mean, var = var)
            else:
                fig, ax = plt.subplots()
            plt.plot(Truth[:,1], Truth[:,0], color='blue', label='Truth')
        else:
            fig, ax = plt.subplots()
            
        if (Plot_Ensemble == 'Ensemble') | (Plot_Ensemble == 'ensemble'):
            for i in range(0, len(Ensemble[0,:])-1):
                plt.plot(Ensemble[:,-1], Ensemble[:,i], color='darkred', linewidth=.25, label="Ensemble Members" if i == 0 else "")
                
        ##### Currently plots 2 separate graphs
        elif Plot_Ensemble == 'PDF':
            M, V = Data_Analysis().Ensemble_Distribution(Ensemble=Ensemble)
            X_lower = M - np.sqrt(V)
            X_upper = M + np.sqrt(V)
            ax.fill_between(Ensemble[:,-1], X_lower, X_upper, color='darkred', label="Ensemble Distribution")
        
        plt.title(f'Ensemble Plot with {len(Ensemble[0,:])-1} Simulations')
        plt.xlabel('Time')
        plt.legend()
        plt.show()
        
    @staticmethod
    def Plot(Data, Style = 'Scatter', mean = 'y', var = 'y'):
        x = Data[:,0]
        t = Data[:,1]
        type(t)
        
        if Style == 'Scatter':
            plt.scatter(t,x)
        if Style == 'Plot':
            plt.plot(t,x)
        if Style == 'Hist':
            bins = int(input('how many bins do you want? '))
            plt.hist(x, bins=bins)
        if Style == 'Box':
            plt.boxplot(x)
        if Style == 'Violin':
            plt.violinplot(x) 
        if Style == 'PDF-Evo':
            if mean == 'y':
                mean = input("Enter the Time Evolution of your SDE's Mean (e.g., 'np.exp(-.2*t)+.5'): ")
            mean = eval(f"lambda t: {mean}")
            if var == 'y':
                var = input("Enter the Time Evolution of your SDE's Variance (e.g., 'np.exp(-.2*t)+.5'): ")
            var = eval(f"lambda t: {var}")
            
            X_lower = mean(t) - np.sqrt(var(t))
            X_upper = mean(t) + np.sqrt(var(t))
            fig, ax = plt.subplots()
            ax.fill_between(t, X_lower, X_upper, color="blue", alpha = .2, label='Distribution')
            plt.plot(t,x)
            
