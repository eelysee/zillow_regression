import pandas as pd
import os
from env import get_connection

def get_tableau_zil_map():
    '''
    map query
    cache's dataframe in a .csv
    from env import get_connection
    '''
    filename = 'tab_zil.csv'
    
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    
    else:
        query ='''
SELECT parcelid, propertylandusetypeid, latitude , longitude , taxvaluedollarcnt
FROM properties_2017
WHERE parcelid IN (
                SELECT parcelid
                FROM predictions_2017
                WHERE LEFT(transactiondate, 4) = '2017'
                )
;
                '''
        
        url = get_connection('zillow')
        df = pd.read_sql(query,url)
        df.to_csv(filename, index=False)
        
        return df

    
def fix_map():
    '''
   
    fixs columns map
    drops nulls and 
    returns a saved csv for import to tablaeu
    '''
    df = get_tableau_zil_map()
    filename = 'tab_zil.csv'
   
    df.dropna(inplace=True)
    df['latitude'] = df['latitude'] / 1e6
    df['longitude'] = df['longitude'] / 1e6
    df.rename(columns={'propertylandusetypeid':'type','longitude':'long', 'latitude':'lat', 'taxvaluedollarcnt':'value'} ,inplace=True)
    df.to_csv(filename, index=False)
    return df


fix_map()
