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
