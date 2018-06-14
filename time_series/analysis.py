from math import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    

def autocorrelate(x,shift=1,conf_int=False,lags=None,df=False):
    
    if isinstance(x,pd.DataFrame) or isinstance(x,pd.Series):
        
        x = x.values
        
    n = len(x)
        
    if lags is None:
        
        lags = n
        
    else:
        
        lags = lags
        
    r_array = np.empty(lags)
    conf_lower = np.empty(lags)
    conf_upper = np.empty(lags)
        
    for i in range(lags):
        
        r_array[i] = acf_compute(x[i:],x[:len(x)-i])
        conf_lower[i] = -1.96 / np.sqrt(len(x)-i)
        conf_upper[i] = 1.96 / np.sqrt(len(x)-i)
        
    if df:
        
        r_array = pd.DataFrame(data=r_array)
        
    if conf_int:
        
        return r_array, conf_upper, conf_lower
    
    return r_array

def plot_auto_corr(x,title=None,lags=None):
    
    auto_corr, conf_upper, conf_lower = autocorrelate(x,conf_int=True,lags=lags)
    
    plt.plot(auto_corr,linestyle='none',marker='o',color='red')

    for i, x in enumerate(auto_corr):
        plt.vlines(x=i,ymin=min(0,x),ymax=max(0,x))
        
    plt.fill_between([i for i in range(len(auto_corr))],conf_lower,conf_upper,color='green',alpha=0.2)
    
    if title is None:
        title = 'Autocorrelation (Lags = {})'.format(len(auto_corr))
        
    else:
        
        title = title + ' (Lags = {})'.format(len(auto_corr))
    plt.title(title,fontsize=16)
    plt.show()

    