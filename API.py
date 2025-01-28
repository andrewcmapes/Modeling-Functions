
import pandas_datareader.data as pdr
import numpy as np
import pandas as pd
import json # JSON is the JavaScript Object Notation library and is used in this code for converting python formats when working with the BLS API
import requests # Requests allows for easy interactions with HTTP web servers/APIs
from datetime import datetime
import sys
import os



main = pd.DataFrame({
        "SeriesID": ['CUUR0000SA0','CUUR0000SAF1','CUUR0000SA0E','CUUR0000SA0L1E'],
        "Names": ['All Items','Food', 'Energy', 'All Items Less Food and Energy'],
        "Weights": [100, 13.495, 6.748, 79.758]
        })



sub = pd.DataFrame({
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



key = '8490e9946b364cf9adcbd4e5f22cb316'
URL = 'https://api.bls.gov/publicAPI/v2/timeseries/data/'



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


###### I want to change this so you can select specific series instead of doing the Tiers
def BLS(Tier = 'Main', Years = [datetime.now().year-1, datetime.now().year]):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    # Selecting data series to be called and converting to a JSON.dumps acceptable format
    if (Tier == 'Main') | (Tier == 'main'):
        Codes = main
    else:
        Codes = sub
    Series = Codes['SeriesID'].to_list()
            
    # Defining the request data format for the API to interpret
    Headers = {'content-type':'application/json'}
    
    # Formatting request data into JSON
    Data = json.dumps({"seriesid": Series, "startyear": Years[0], "endyear": Years[1], 'registrationKey':key})
    
    # Sending request to API, then reading it as JSON data
    Response = requests.post(URL, data = Data, headers = Headers).json()
    
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
