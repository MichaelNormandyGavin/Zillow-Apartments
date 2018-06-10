from math import sqrt
import numpy as np
import pandas as pd



def compute_correlation(x,y,r2=False,auto=False):
    
    '''Take two array-like series to calculate the correlation
    x: numpy.array or pandas.DataFrame: x value for correlation
    y: numpy.array or pandas.DataFrame: y value for correlation
    
    r2: Boolean (optional): return r-squared value instead of r'''
    
    '''Need to remove the mean for autocorrelation?'''
    
    df = pd.DataFrame({'x':x,'y':y})
    
    if auto:
        
        df['x'] = df['x'] - df['x'].mean()
        df['y'] = df['y'] - df['y'].mean()
    
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
    
    #corr_test = np.cov(df['x'].values,df['y'].values)[0,1]
    
    return df, corr

def acf_compute(x,y):
    
    if isinstance(x,pd.DataFrame) or isinstance(x,pd.Series):
        
        x = x.dropna().values
        
    if isinstance(y,pd.DataFrame) or isinstance(y,pd.Series):
        
        y = y.dropna().values
           
    nx = len(x)
    ny = len(y)
    
    x = x[nx-ny:]
    
    top = np.mean(np.dot((x-np.mean(x)), (y-np.mean(y))))
    
    bot = np.sum(np.square((x-np.mean(x))))
    
    acf_r = top/bot
    
    return acf_r
    

def autocorrelate(x,shift=1,lags=None,df=False):
    
    if isinstance(x,pd.DataFrame) or isinstance(x,pd.Series):
        
        x = x.values
        
    if lags is None:
        
        lags = len(x)
        
    else:
        
        lags = lags
        
    r_array = np.empty(lags)
        
    for i in range(lags):
        
        r_array[i] = acf_compute(x[i:],x[:len(x)-i])
        
    if df:
        
        r_array = pd.DataFrame(data=r_array)
    
    return r_array