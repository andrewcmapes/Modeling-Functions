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
