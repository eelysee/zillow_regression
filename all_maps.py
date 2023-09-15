import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import os
from env import get_connection
from PIL import Image


def get_zillow_map():
    '''
    map query
    
    cache's dataframe in a .csv
    from env import get_connection
    '''
    filename = 'map.csv'
    
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    
    else:
        query ='''
SELECT propertylandusetypeid, latitude , longitude , taxvaluedollarcnt
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

    
def fix_map(df):
    '''
    fixs columns map
    drops nulls and 
    '''
    df.dropna(inplace=True)
    df['latitude'] = df['latitude'] / 1e6
    df['longitude'] = df['longitude'] / 1e6
    df.rename(columns={'propertylandusetypeid':'type','longitude':'long', 'latitude':'lat', 'taxvaluedollarcnt':'value'} ,inplace=True)
    return df


def plotly_zill():
# Plotting
    fig = px.scatter_geo(
        df, 
        lat='latitude', 
        lon='longitude',
        color='taxvaluedollarcnt', 
        projection="natural earth",
        title="Property Locations and Values",
        color_continuous_scale=px.colors.sequential.Plasma
    )
    fig.update_geos(showcoastlines=True, coastlinecolor="Black", showland=True, landcolor="grey")
    fig.update_layout(coloraxis_colorbar=dict(title="Property Value"))
    fig.show()
    
def zil_map_img():
    '''
    pulls in images and plots them. 
    images already saved to save folder with corresponding names.
    '''
    legend = Image.open('legend.png')
    california = Image.open('california.png')
    streets = Image.open('Streets.png')
    streets_la = Image.open('streets_la.png')
    streets_ventura = Image.open('streets_ventura.png')
    
    plt.figure()
    
    plt.imshow(california)
    plt.title('California')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
    plt.imshow(legend)
    plt.title('Legend')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
    plt.imshow(streets)
    plt.title('Streets')
    plt.xticks([])
    plt.yticks([])
    plt.show()
  
    plt.imshow(streets_la)
    plt.title('Streets LA')
    plt.xticks([])
    plt.yticks([])
    plt.show()

    plt.imshow(streets_ventura)
    plt.title('Streets Ventura')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
