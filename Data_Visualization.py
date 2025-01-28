from data_analysis import ensemble_distribution
import matplotlib.pyplot as plt
import numpy as np



def ensemble_plot(ensemble, truth = None, plot_ensemble = 'ensemble', plot_truth = 'PDF-evo', mean = 'y', var = 'y'):
    if truth.any() != None:
        if plot_truth == 'PDF-Evo':
            plot(truth, style = 'PDF-Evo', mean = mean, var = var)
        else:
            fig, ax = plt.subplots()
        plt.plot(truth[:,1], truth[:,0], color='blue', label='truth')
    else:
        fig, ax = plt.subplots()
        
    if (plot_ensemble == 'ensemble') | (plot_ensemble == 'ensemble'):
        for i in range(0, len(ensemble[0,:])-1):
            plt.plot(ensemble[:,-1], ensemble[:,i], color='darkred', linewidth=.25, label="ensemble members" if i == 0 else "")
            
    ##### Currently plots 2 separate graphs
    elif plot_ensemble == 'PDF':
        m, v = ensemble_distribution(ensemble=ensemble)
        x_lower = m - np.sqrt(v)
        x_upper = m + np.sqrt(v)
        ax.fill_between(ensemble[:,-1], x_lower, x_upper, color='darkred', label="ensemble Distribution")
    
    plt.title(f'ensemble plot with {len(ensemble[0,:])-1} Simulations')
    plt.xlabel('Time')
    plt.legend()
    plt.show()


def plot(data, style = 'scatter', mean = 'y', var = 'y'):
    x = data[:,0]
    t = data[:,1]
    type(t)
    
    if style == 'scatter':
        plt.scatter(t,x)
    if style == 'plot':
        plt.plot(t,x)
    if style == 'hist':
        bins = int(input('how many bins do you want? '))
        plt.hist(x, bins=bins)
    if style == 'box':
        plt.boxplot(x)
    if style == 'violin':
        plt.violinplot(x) 
    if style == 'PDF-evo':
        if mean == 'y':
            mean = input("Enter the Time Evolution of your SDE's mean (e.g., 'np.exp(-.2*t)+.5'): ")
        mean = eval(f"lambda t: {mean}")
        if var == 'y':
            var = input("Enter the Time Evolution of your SDE's variance (e.g., 'np.exp(-.2*t)+.5'): ")
        var = eval(f"lambda t: {var}")
        
        x_lower = mean(t) - np.sqrt(var(t))
        x_upper = mean(t) + np.sqrt(var(t))
        fig, ax = plt.subplots()
        ax.fill_between(t, x_lower, x_upper, color="blue", alpha = .2, label='Distribution')
        plt.plot(t,x)
