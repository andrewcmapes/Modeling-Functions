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



class Attributes:
    def __init__(self):
        self.Main = pd.DataFrame({
            "SeriesID": ['CUUR0000SA0','CUUR0000SAF1','CUUR0000SA0E','CUUR0000SA0L1E'],
            "Names": ['All Items','Food', 'Energy', 'All Items Less Food and Energy'],
            "Weights": [100, 13.495, 6.748, 79.758]
            })
            
        self.Sub = pd.DataFrame({
            "SeriesID": [
                'CUUR0000SA0','CUUR0000SAF11','CUUR0000SEFV','CUUR0000SETB01','CUUR0000SEHF01',
                'CUUR0000SEHF02','CUUR0000SAA','CUUR0000SETA01','CUUR0000SETA02','CUUR0000SAM1',
                'CUUR0000SAF116','CUUR0000SEGA','CUUR0000SEHA','CUUR0000SEHC','CUUR0000SEMC01',
                'CUUR0000SEMD','CUUR0000SETD','CUUR0000SETE','CUUR0000SETG01'
                ],
            "Names": [
                'All Items','Food at Home','Food Away from Home', 'Gas', 'Electricity', 'Utility Gas',
                'Apparel', 'New Vehicles', 'Used Vehicles', 'Medical Care Commodities', 'Alcohol',
                'Tobacco', 'Rent of Primary Residence', 'Owners Equivalent Rent', 'Physicians Services',
                'Hospital and Related Services','Vehicle Maintenance', 'Car Insurance', 'Air Fares'
                ],
            "Weights": [
                100, 8.138, 5.356, 3.312, 2.464, 0.695, 2.605, 3.648, 1.921, 1.464,
                0.848, 0.541, 7.639, 26.713, 1.814, 1.983, 1.234, 2.854, 0.806
                ]
            })
        
        self.Key = '8490e9946b364cf9adcbd4e5f22cb316'
        self.URL = 'https://api.bls.gov/publicAPI/v2/timeseries/data/'

class Data_Generator:
    @staticmethod
    def Forward_Euler(deterministic = 'y', dt = 0.01, T = 50, X_0 = 0):
        if deterministic == 'y':
            deterministic = input("Enter the deterministic portion of your SDE (e.g., 'x**2 + 2'): ")
        deterministic = eval(f"lambda x: {deterministic}")
        
        steps = int(np.round(T/dt)) + 1
        X = np.zeros((steps, 3))
        X[0, 0] = X_0
        
        for i in range(0,steps-1):
            X[i+1,0] = X[i,0] + deterministic(X[i,0]) * dt
            X[i+1,1] = X[i,1] + dt
        return X

    @staticmethod
    def Euler_Maruyama(deterministic = 'y', noise = 'y', dt = 0.01, T = 50, X_0 = 0):
        if deterministic == 'y':
            deterministic = input("Enter the deterministic portion of your SDE (e.g., 'x**2 + 2'): ")
        if "t" in deterministic:
            deterministic = eval(f"lambda x, t: {deterministic}")
        else:
            deterministic = eval(f"lambda x: {deterministic}")
            
        if noise == 'y':
            noise = input("Enter the stochastic portion of your SDE (e.g., 'x**2 + 2'): ")
        if "t" in noise:
            noise = eval(f"lambda x, t: {noise}")
        else:
            noise = eval(f"lambda x: {noise}")
        
        steps = int(np.round(T/dt)) + 1
        X = np.zeros((steps, 3))
        X[0, 0] = X_0
        
        for i in range(0,steps-1):
            X[i+1,1] = X[i,1] + dt
            X[i+1,2] = np.sqrt(dt) * np.random.normal(0,1)
            if "t" in inspect.signature(deterministic).parameters:
                det = deterministic(X[i,0], X[i,1])
            else:
                det = deterministic(X[i,0])
            if "t" in inspect.signature(noise).parameters:
                sto = noise(X[i,0], X[i,1])
            else:
                sto = noise(X[i,0])
            X[i+1,0] = X[i,0] + det * dt + sto * X[i,2]
            
        return X
    
    @staticmethod
    def Milstein(deterministic = 'y', noise = 'y', diff_noise = 'y', dt = 0.01, T = 50, X_0 = 0):
        if deterministic == 'y':
            deterministic = input("Enter the deterministic portion of your SDE (e.g., 'x**2 + 2'): ")
        deterministic = eval(f"lambda x: {deterministic}")
        if noise == 'y':
            noise = input("Enter the stochastic portion of your SDE (e.g., 'x**2 + 2'): ")
        noise = eval(f"lambda x: {noise}")
        if diff_noise == 'y':
            diff_noise = input("Enter the stochastic portion of your SDE's derivative (e.g., 'x**2 + 2'): ")
        diff_noise = eval(f"lambda x: {diff_noise}")
        
        steps = int(np.round(T/dt)) + 1
        X = np.zeros((steps, 3))
        X[0, 0] = X_0
        
        for i in range(0,steps-1):
            X[i+1,2] = np.sqrt(dt) * np.random.normal(0,1)
            X[i+1,0] = X[i,0] + deterministic(X[i,0]) * dt + noise(X[i,0]) * X[i,2] + (1/2) * noise(X[i,0]) * diff_noise(X[i,0]) * (X[i,2]**2-dt**2)
            X[i+1,1] = X[i,1] + dt
            
        return X
    
    @staticmethod
    def Ensemble(Type = 'EM', N = 10, deterministic = 'y', noise = 'y', diff_noise = 'y', dt = 0.01, T = 50, X_0 = 0):
        n = int(T/dt) + 1
        if Type == 'FE':
            if deterministic == 'y':
                deterministic = input("Enter the deterministic portion of your SDE (e.g., 'x**2 + 2'): ")
            Data = np.tile([0.0]*(N+1), (n,1))
            for i in range(0,N):
                X = Data_Generator.Forward_Euler(deterministic = deterministic, dt = dt, T = T, X_0 = X_0)
                Data[:,i] = X[:,0]
            Data[:,N] = X[:,1]
        
        elif Type == 'EM':
            if deterministic == 'y':
                deterministic = input("Enter the deterministic portion of your SDE (e.g., 'x**2 + 2'): ")
            if noise == 'y':
                noise = input("Enter the stochastic portion of your SDE (e.g., 'x**2 + 2'): ")
            Data = np.tile([0.0]*(N+1), (n,1))
            for i in range(0,N):
                X = Data_Generator.Euler_Maruyama(deterministic = deterministic, noise = noise, dt = dt, T = T, X_0 = X_0)
                Data[:,i] = X[:,0]
            Data[:,N] = X[:,1]
            
        return Data
    
    @staticmethod
    def Ordinary_Equation(equation = 'y', noise = 'y', dt = 0.01, T = 50, X_0 = 0):
        if equation == 'y':
            equation = input("Enter the Ordinary Equation (e.g., 'x**2 + 2'): ")
        else:
            equation = eval(f"lambda x: {equation}")
            
        if noise == 'y':
            noise = input("Enter the Noise Equation (e.g., 'x**2 + 2'): ")
        else:
            noise = eval(f"lambda x: {noise}")
        
        steps = int(np.round(T/dt)) + 1
        X = np.zeros((steps, 3))
        X[0, 0] = X_0
        Time = 0
        for i in range(0,steps-1):
            Time += dt
            X[i,1] = Time
            X[i,2] = np.sqrt(dt) * np.random.normal(0,1)
            X[i,0] = equation(Time) + noise(equation(Time)) * X[i,2]
        return  X
    
    @staticmethod
    def Random_Poly_SDE(n, Terms='random'):
        return 'in progress'
    
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
            
class API(Attributes):
    def __init__(self):
        super().__init__()
        
        
    ###### I want to change this so you can select specific series instead of doing the Tiers
    def BLS(self, Tier = 'Main', Years = [datetime.now().year-1, datetime.now().year]):
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        # Selecting data series to be called and converting to a JSON.dumps acceptable format
        if (Tier == 'Main') | (Tier == 'main'):
            Codes = self.Main
        else:
            Codes = self.Sub
        Series = Codes['SeriesID'].to_list()
                
        # Defining the request data format for the API to interpret
        Headers = {'content-type':'application/json'}
        
        # Formatting request data into JSON
        Data = json.dumps({"seriesid": Series, "startyear": Years[0], "endyear": Years[1], 'registrationKey':self.Key})
        
        # Sending request to API, then reading it as JSON data
        Response = requests.post(self.URL, data = Data, headers = Headers).json()
        
        # Constructs a DataFrame from the resulting JSON data 
        DataFrame = pd.json_normalize(Response['Results']['series'], record_path=['data'], meta=['seriesID'], errors='ignore')
        
        # Formatting date representation 
        DataFrame['Date'] = DataFrame['year'].astype(str).str[-2:] + DataFrame['period']
        
        # Inverting dataframe structure
        
        Pivot = DataFrame.pivot(index='Date', columns='seriesID', values='value')
        Mapping = dict(Codes[['SeriesID', 'Names']].values)
        Pivot = Pivot.rename(columns=Mapping)
        Pivot.columns.name, Pivot.index.name = None, None
        return Pivot
    
    @staticmethod
    def FRED(Series = 'Standard', Dates = ['',''], Format = 'Standardized'):
        
        # Gathers series IDs to be requested
        if Series == 'Standard':
            Series = []
            more = 'y'
            while more == 'y':
                addition =  input('What data set do you want? Example: GDP  ')
                more = input('Do you have more data sets you want to add? (y/n)  ')
                Series.append(addition)
                
        # Formats dates correctly for data request
        if Dates == ['','']:
            Dates[0] = input('What is your desired start date? Format yyyymmdd  ')
            Dates[1] = input('What is your desired end date? Format yyyymmdd  ')
            
        else:
            if not isinstance(Dates[0], str):
                print('Input Error: Dates must be in string format of "yyyymmdd"')
                sys.exit(3)
        DataFrame = pdr.DataReader(Series, 'fred', start = datetime(int(Dates[0][:4]), int(Dates[0][5:6]), int(Dates[0][7:8])), end = datetime(int(Dates[1][:4]), int(Dates[1][4:6]), int(Dates[1][6:8])))
        DataFrame.index.name = None
        
        # This either returns the data as a single dataframe or converts to match formatting used in other classes
        if Format == 'Standardized':
            Data_Series = {}
            for i in range(0,len(DataFrame.iloc[0,:])):
                Data = DataFrame.iloc[:,i].dropna().reset_index().to_numpy()
                Data[:, [0,1]] = Data[:, [1,0]]
                Data_Series[Series[i]] = Data
            return Data_Series
        else:
            return DataFrame

class VIX:
    def Retriever(month, year, output_dir=r"C:\Users\andre\spyder-env\Projects\Investment Algos\VIX\VIX Data"):
        vix_dictionary = {
            "F": "Jan",
            "G": "Feb",
            "H": "Mar",
            "J": "Apr",
            "K": "May",
            "M": "Jun",
            "N": "Jul",
            "Q": "Aug",
            "U": "Sep",
            "V": "Oct",
            "X": "Nov",
            "Z": "Dec"
            }
        month_dictionary = {
            "Jan": "01",
            "Feb": "02",
            "Mar": "03",
            "Apr": "04",
            "May": "05",
            "Jun": "06",
            "Jul": "07",
            "Aug": "08",
            "Sep": "09",
            "Oct": "10",
            "Nov": "11",
            "Dec": "12"
            }
        base_url = 'https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/VX/'
        days = range(1,32)
        for day in days:
            try:
                # Format month and day to ensure two digits
                m = f"{month:02d}"
                d = f"{day:02d}"

                # Construct filename and URL
                filename = f"VX_{year}-{m}-{d}.csv"
                url = f"{base_url}{filename}"
                
                # Make request to URL
                response = requests.get(url)
                
                # Check if the request was successful
                if response.status_code == 200:
                    # Convert response text to a pandas DataFrame
                    try:
                        data = pd.read_csv(StringIO(response.text))
                        if len(data.iloc[:,0]) < 50:
                            print(f"{filename} was weekly VIX or ")
                            continue
                        elif len(data.iloc[:,0]) >=50:
                            Month = vix_dictionary.get(data.iloc[0,1][0])
                            Year = data.iloc[0,1][-3:-1]
                            del data['Open'], data['Total Volume'], data['EFP'], data['Open Interest'], data['Settle'], data['Change'], data['Close'], data['Low'], data['Futures']
                            data.rename(columns={'High':Month+Year}, inplace=True)
                        
                            # Generate the output CSV filename
                            filename = 'VX_'+Year+'_'+month_dictionary.get(Month)+'.csv'
                            output_file = os.path.join(output_dir, filename)

                            # Save DataFrame to a CSV file
                            data.to_csv(output_file, index=False)
                            print(f"Converted {filename}")

                    except Exception as e:
                        print(f"Failed to process data for {filename}: {e}")

            except Exception as e:
                print(f"Error fetching data for {year}-{m}-{d}: {e}")
 
    def Update():
        date = datetime.now().date()
        years = range(date.year, date.year+2)
        months = range(date.month, 13)

        for year in years:
            for month in months:
                VIX.Retriever(month, year)

    def Spreads(mo_spread=1, differential=True, save=True):
        folder = r"C:\Users\andre\spyder-env\Projects\VIX\VIX Processed Data"
        input_file = 'Merged.csv'
        data = pd.read_csv(os.path.join(folder, input_file))

        spreads = []
        for i in range(0, len(data.iloc[0,:])-10):
            spread = (data.iloc[:,i+mo_spread]-data.iloc[:,i]).dropna().reset_index(drop=True)
            spreads.append(spread[::-1].reset_index(drop=True)[:(21*(8-mo_spread))])
        
        table = pd.concat(spreads,axis=1).reset_index(drop=True)
        
        if save == True:
            output_file = f'{mo_spread} Month Spreads.csv'
            table.to_csv(os.path.join(folder, output_file), index=False)
            print(f'Successfully saved {output_file}')
        
        if differential == True:
            differentials = []
            for i in range(0, len(table.iloc[:,0])-1):
                differential = table.iloc[i,:] - table.iloc[i+1,:]
                differentials.append(differential)
            table2 = pd.DataFrame(differentials).round(2)
            output_file = f'{mo_spread} Month Differentials.csv'
            table2.to_csv(os.path.join(folder, output_file), index=False)
            print(f'Successfully saved {output_file}')
            return table, table2
        
        return table
    