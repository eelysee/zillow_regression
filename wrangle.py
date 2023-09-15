import pandas as pd
import numpy as np
import os

from env import get_connection
from sklearn.model_selection import train_test_split

# 
def get_zillow_mvp():
    '''
    MVP query
    
    cache's dataframe in a .csv
    '''
    filename = 'mvp.csv'
    
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    
    else:
        query ='''
               SELECT bathroomcnt, bedroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt
FROM properties_2017
WHERE propertylandusetypeid = (
                                SELECT propertylandusetypeid
                                FROM propertylandusetype
                                WHERE propertylandusedesc = 'Single Family Residential'
                                )
AND parcelid IN (
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

    
  
def drop_zill_mvp(zillow):    
    '''
    Dropping 1 null values
    bed 7 = 6 plus
    bath 6 = 5.5+
    rename columns 
    '''
    zillow.dropna(inplace= True)
    zillow.rename(columns = {'bedroomcnt':'bed', 'bathroomcnt':'bath', 'calculatedfinishedsquarefeet': 'sqft', 'taxvaluedollarcnt': 'value'}, inplace=True)
    zillow['bath'] = np.where(zillow['bath'] >= 5.5, '6', zillow['bath'])
    zillow['bed'] = np.where(zillow['bed'] >= 7, '8', zillow['bed'])
    
    return zillow



def train_val_test(df, seed = 55):
    '''
    splits to train val test
    TAKES 1 df
    '''
    train, val_test = train_test_split(df, train_size = 0.7,
                                       random_state = seed)
    
    val, test = train_test_split(val_test, train_size = 0.5,
                                 random_state = seed)
    
    return train, val, test


def X_y_split(train, val, target):
    '''
    Splits train and val into X and Y splits for target testing.
    
    target is target variable entered as the name of the column only in quotes 
    
    returns X_train, y_train , X_val , y_val
    '''
    t = target
    X_train = train.drop(columns=[t])
    y_train = train[t]
    X_val = val.drop(columns=[t])
    y_val = val[t]
    return X_train, y_train , X_val , y_val
