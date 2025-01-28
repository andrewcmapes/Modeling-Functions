
import pandas_datareader.data as pdr
import numpy as np
import pandas as pd
import json # JSON is the JavaScript Object Notation library and is used in this code for converting python formats when working with the BLS API
import requests # Requests allows for easy interactions with HTTP web servers/APIs
from datetime import datetime
import sys
import os



main = pd.DataFrame({
        "seriesid": ['CUUR0000SA0','CUUR0000SAF1','CUUR0000SA0E','CUUR0000SA0L1E'],
        "names": ['All Items','Food', 'Energy', 'All Items Less Food and Energy'],
        "weights": [100, 13.495, 6.748, 79.758]
        })



sub = pd.DataFrame({
        "seriesid": [
            'CUUR0000SA0','CUUR0000SAF11','CUUR0000SEFV','CUUR0000SETB01','CUUR0000SEHF01',
            'CUUR0000SEHF02','CUUR0000SAA','CUUR0000SETA01','CUUR0000SETA02','CUUR0000SAM1',
            'CUUR0000SAF116','CUUR0000SEGA','CUUR0000SEHA','CUUR0000SEHC','CUUR0000SEMC01',
            'CUUR0000SEMD','CUUR0000SETD','CUUR0000SETE','CUUR0000SETG01'
            ],
        "names": [
            'All Items','Food at Home','Food Away from Home', 'Gas', 'Electricity', 'Utility Gas',
            'Apparel', 'New Vehicles', 'Used Vehicles', 'Medical Care Commodities', 'Alcohol',
            'Tobacco', 'Rent of Primary Residence', 'Owners Equivalent Rent', 'Physicians Services',
            'Hospital and Related Services','Vehicle Maintenance', 'Car Insurance', 'Air Fares'
            ],
        "weights": [
            100, 8.138, 5.356, 3.312, 2.464, 0.695, 2.605, 3.648, 1.921, 1.464,
            0.848, 0.541, 7.639, 26.713, 1.814, 1.983, 1.234, 2.854, 0.806
            ]
        })



key = '8490e9946b364cf9adcbd4e5f22cb316'
url = 'https://api.bls.gov/publicAPI/v2/timeseries/data/'



def fred(series = 'standard', dates = ['',''], format = 'standardized'):
    # Gathers series IDs to be requested
    if series == 'standard':
        series = []
        more = 'y'
        while more == 'y':
            addition =  input('What data set do you want? Example: GDP  ')
            more = input('Do you have more data sets you want to add? (y/n)  ')
            series.append(addition)
            
    # Formats dates correctly for data request
    if dates == ['','']:
        dates[0] = input('What is your desired start date? Format yyyymmdd  ')
        dates[1] = input('What is your desired end date? Format yyyymmdd  ')
        
    else:
        if not isinstance(dates[0], str):
            print('Input Error: dates must be in string format of "yyyymmdd"')
            sys.exit(3)
    df = pdr.DataReader(series, 'fred', start = datetime(int(dates[0][:4]), int(dates[0][5:6]), int(dates[0][7:8])), end = datetime(int(dates[1][:4]), int(dates[1][4:6]), int(dates[1][6:8])))
    df.index.name = None
    
    # This either returns the data as a single dataframe or converts to match formatting used in other classes
    if format == 'standardized':
        data_series = {}
        for i in range(0,len(df.iloc[0,:])):
            data = df.iloc[:,i].dropna().reset_index().to_numpy()
            data[:, [0,1]] = data[:, [1,0]]
            data_series[series[i]] = data
        return data_series
    else:
        return df


###### I want to change this so you can select specific series instead of doing the tiers
def bls(tier = 'main', years = [datetime.now().year-1, datetime.now().year]):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    # Selecting data series to be called and converting to a JSON.dumps acceptable format
    if (tier == 'main') | (tier == 'main'):
        codes = main
    else:
        codes = sub
    series = codes['seriesid'].to_list()
            
    # Defining the request data format for the API to interpret
    headers = {'content-type':'application/json'}
    
    # Formatting request data into JSON
    data = json.dumps({"seriesid": series, "startyear": years[0], "endyear": years[1], 'registrationKey':key})
    
    # Sending request to API, then reading it as JSON data
    response = requests.post(url, data = data, headers = headers).json()
    
    # Constructs a DataFrame from the resulting JSON data 
    df = pd.json_normalize(response['Results']['series'], record_path=['data'], meta=['seriesID'], errors='ignore')
    
    # Formatting date representation 
    df['Date'] = df['year'].astype(str).str[-2:] + df['period']
    
    # Inverting dataframe structure
    
    pivot = df.pivot(index='Date', columns='seriesID', values='value')
    mapping = dict(codes[['seriesid', 'names']].values)
    pivot = pivot.rename(columns=mapping)
    pivot.columns.name, pivot.index.name = None, None
    return pivot
