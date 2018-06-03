from math import sqrt
import numpy as np
import pandas as pd



def compute_correlation(x,y,r2=False):
    
    '''Take two array-like series to calculate the correlation
    x: numpy.array or pandas.DataFrame: x value for correlation
    y: numpy.array or pandas.DataFrame: y value for correlation
    
    r2: Boolean (optional): return r-squared value instead of r'''
    
    
    
    df = pd.DataFrame({'x':x,'y':y})
    
    df.dropna(inplace=True)
    
    n = len(df)
    
    df['x2'] = np.square(df['x'])
    df['y2'] = np.square(df['y'])
    df['xy'] = df['x'] * df['y']
    sum_x = df['x'].sum()
    sum_y = df['y'].sum()
    sum_xy = df['xy'].sum()
    sum_x2 = df['x2'].sum()
    sum_y2 = df['y2'].sum()
    
    corr = (n*(sum_xy) - (sum_x*sum_y)) / (sqrt(((n*(sum_x2) - (sum_x**2)) *((n*(sum_y2) - (sum_y**2))))))
    
    return df, corr

def autocorrelate(x,shift=1,**kwargs):
    
    df, corr = compute_correlation(x, x.shift(-shift),**kwargs)
    
    return df, corr