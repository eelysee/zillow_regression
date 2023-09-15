import pandas as pd
from env import get_connection
import os



def get_zillow_full():
    '''
    full 2017 properties table
    '''
    filename = 'properties_2017.csv'
   
    query ='''
                SELECT *
                FROM properties_2017
                ;
                '''
        
    url = get_connection('zillow')
    df = pd.read_sql(query,url)
    df.to_csv(filename, index=False)
        
    
get_zillow_full()