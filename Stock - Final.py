#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 18:53:27 2019

@author: marinashah
"""
from pylab import plot,show
from numpy import vstack,array
from numpy.random import rand
import numpy as np
from scipy.cluster.vq import kmeans,vq
import pandas as pd
import pandas_datareader as dr
from math import sqrt
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import string
from selenium import webdriver
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.piecewise import PiecewiseAggregateApproximation
from tslearn.piecewise import SymbolicAggregateApproximation, OneD_SymbolicAggregateApproximation

# =============================================================================
# Part 1:
# Data acquisition  
# =============================================================================

urlTheStar='https://www.thestar.com.my/business/marketwatch/stock-list/?alphabet='
alpha = []
for letter in string.ascii_uppercase:
    alpha.append(letter)     
alpha.append('0-9')
print("!!!  Array of chars")
print(alpha)

stockname = []
for i in alpha:
    print("!!!  Now char "+ i)
    browser = webdriver.Firefox(executable_path='/Users/marinashah/Desktop/Data Mining/geckodriver')
    browser.implicitly_wait(40)
    browser.get(urlTheStar + i)
    WebDriverWait(browser,40).until(EC.visibility_of_element_located((By.ID,'marketwatchtable')))
    innerHTML = browser.find_element_by_id("marketwatchtable").get_attribute("innerHTML")
    soup = BeautifulSoup(innerHTML, 'lxml') 
    links = soup.findAll('a')
    for link in links:
        splitlink = link['href'].split('=')
        stock = splitlink[1]
        stockname.append(stock)
        print(stock)
    browser.close()

dict = {'name':stockname}
df_stockname = pd.DataFrame(dict)
df_stockname.to_csv('stockname.csv')

df = pd.read_csv('price_df.csv',usecols=[i for i in range(8) if i != 0]);print(df)

# =============================================================================
# Part 2:
# Exploratory data analysis
# =============================================================================

# find the change in daily closing prices
df['dif'] = df.groupby('name')['volume'].diff()
print(df)
df.to_csv('df_volume.csv')

#transpose data frame according to name, and indexing by date for closing prices only
df_transposed = df.set_index(['name','day']).close.unstack('name')
print(df_transposed)
df_transposed.to_csv('volume_transposed.csv')

#compute stock daily return/change
df_return = df_transposed.pct_change()
print(df_return)
df_return.to_csv('volume_transpose_pct_change.csv')
#can use transpose_pct_change to do regression in SAS. 

df_return['ASIAPAC-WB']
# create covariance matrix
df_cov = df_return.cov()
df_cov.to_csv('volume_cov.csv')
# create correlation matrix
df_corr = df_return.corr()
df_corr.to_csv('volume_corr.csv')

# variation of number of records per stock
groups = df.groupby(['name'])
groups.get_group('AIRPORT-C8') #only have 1
groups.get_group('3A') # have full records for 3 months


#for more meaningful covariance interpretation, we decided to filter stocks with 60 records or more
df_new = groups.filter(lambda x : len(x)>=60)
df_new.to_csv('volume_stocks60.csv')
df_transposed1 = df_new.set_index(['name','day']).close.unstack('name')
df_return1 = df_transposed1.pct_change()
df_return1.head()
df_return1.to_csv('volume_transpose_pct_change1.csv')
df_return2 = df_return1.dropna(how='any',axis=0)
df_return2.to_csv('volume_transpose_pct_change2.csv')
print(df_return2)
 

df_return1.cov()
df_return1.to_csv('volume_cov60.csv')
df_return1.corr()
df_return1.to_csv('volume_cor.csv')


#Calculate average annual percentage return and volatilities over a theoretical one year period
returns = df_return2.mean() * 61
returns = pd.DataFrame(returns)
print(returns)
returns.to_csv('volume_return_pct.csv')
returns.columns = ['Returns']
returns['Volatility'] = df_return2.std() * sqrt(61)
print(returns)
returns.to_csv('volume_return_volatility.csv') 


# =============================================================================
# Part 3:
# K Mean Clustering Return against Volatility 
# =============================================================================

data = np.asarray([np.asarray(returns['Returns']),np.asarray(returns['Volatility'])]).T

print(data)
 
X = data
distorsions = []
for k in range(2, 20):
    k_means = KMeans(n_clusters=k)
    k_means.fit(X)
    distorsions.append(k_means.inertia_)
 
fig = plt.figure(figsize=(15, 5))
plt.plot(range(2, 20), distorsions)
plt.grid(True)
plt.title('Elbow curve')
plt.xlabel("Number of cluster")
plt.ylabel("WCSS")
plt.savefig('Volume_Elbow Curve_test.png')


#===============5 clusters=====================#
# computing K-Means with K = 5 (5 clusters)
centroids,_ = kmeans(data,5)
# assign each sample to a cluster
idx,_ = vq(data,centroids)
 
# some plotting using numpy's logical indexing
plt.plot(data[idx==0,0],data[idx==0,1],'ob',
     data[idx==1,0],data[idx==1,1],'oy',
     data[idx==2,0],data[idx==2,1],'or',
     data[idx==3,0],data[idx==3,1],'og',
     data[idx==4,0],data[idx==4,1],'om')
plt.plot(centroids[:,0],centroids[:,1],'sg',markersize=8)
plt.title ("Clusters of Stocks")
plt.xlabel("Return")
plt.ylabel("Volatility")
plt.savefig('Cluster_test, K=5.png')
plt.show()

#===============5 clusters details=================================#
details = [(name,cluster) for name, cluster in zip(returns.index,idx)]
details = pd.DataFrame(details)
details.to_csv('details_5cluster.csv')

#to sort for only cluster 2# (VOLATILE CLUSTER)
details.rename(columns = {details.columns[0]: 'name', details.columns[1]: 'cluster'}, inplace = True)
details.head()

details = details.sort_values('cluster')
details['cluster'].value_counts().sort_index()

details.set_index('cluster', inplace=True)

#create dataframe for cluster 3 ###############3nnt test balil==========
clusters_high = details.loc[3]
clusters_high = pd.DataFrame(clusters_high)
clusters_high_name = clusters_high['name']
print(clusters_high_name)


#=============Cluster frequency===========================#
menMeans = [137,69,27,11,243]
ind = ['0', '1', '2','3','4']

fig, ax = plt.subplots(figsize = (10,5))
ax.bar(ind,menMeans,width=1.0, color=['blue', 'yellow', 'red', 'green', 'magenta'])
ax.set_ylabel('Frequency')
ax.set_xlabel('Cluster')
ax.set_title('Stock Cluster Details')

for index,data in enumerate(menMeans):
    plt.text(x=index , y =data+1 , s=f"{data}" , fontdict=dict(fontsize=14))
plt.savefig('Cluster Details.png')
plt.show()

#recall Returns and Volatility reading based on cluster 3
Returns_Vol_Re = returns.loc[['FGV-C63', 'ORION-WA', 'SUMATEC','NETX', 'TRIVE', \
                              'ZELAN','DAYA', 'HSI-C3Z', 'HSI-C5A', 'HSI-C5B',\
                              'DBHD-WA'],['Returns','Volatility']]

Returns_Vol_Re = pd.DataFrame(Returns_Vol_Re)
print(Returns_Vol_Re)

#merge cluster 2 to return & volatility
Cluster_Return_Vol = pd.merge(clusters_high, Returns_Vol_Re, how='inner', on='name')
Cluster_Return_Vol.sort_values(by=['Returns'], inplace=True, ascending=False)
print(Cluster_Return_Vol)
Cluster_Return_Vol.to_csv('volume_Cluster_Return_Vol.csv')


#recall all close price for the selected share.
df_new = groups.filter(lambda x : len(x)>=60)
df_new.head()
df_new.columns
df_new.set_index('name', inplace=True)
volume_price = df_new.loc[['FGV-C63', 'ORION-WA', 'SUMATEC','NETX', 'TRIVE', \
                              'ZELAN','DAYA', 'HSI-C3Z', 'HSI-C5A', 'HSI-C5B',\
                              'DBHD-WA'],\
                        ['day','close','open', 'high', 'low','volume','dif']]
volume_price.head()
volume_price.to_csv('volume_price.csv')
volume_price = volume_price.reset_index()
volume_price.head()

df_volume_transpose = volume_price.set_index(['name','day']).volume.unstack('name')
df_volume_transpose.head()
df_volume_transposed_pct = df_volume_transpose.pct_change()
df_volume_transposed_pct2 = df_volume_transposed_pct.dropna(how='any',axis=0)
df_volume_transposed_pct2.head()

df_volume_transposed_pct2.cov()
df_volume_transposed_pct2.to_csv('volume_cov_close.csv')
df_volume_transposed_pct2.corr()
df_volume_transposed_pct2.to_csv('volume_corr_close.csv')

# function defined to compute top positive and negative correlation
#=================================================================================
def get_redundant_pairs(df_volume_transposed_pct2):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df_volume_transposed_pct2.columns
    for i in range(0, df_volume_transposed_pct2.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_pos_correlations(df_volume_transposed_pct2, n):
    au_corr = df_volume_transposed_pct2.corr().unstack()
    labels_to_drop = get_redundant_pairs(df_volume_transposed_pct2)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

def get_top_neg_correlations(df_volume_transposed_pct2, n):
    au_corr = df_volume_transposed_pct2.corr().unstack()
    labels_to_drop = get_redundant_pairs(df_volume_transposed_pct2)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=True)
    return au_corr[0:n]
#=================================Top 10 correlations===============================

n = 10

#For top n positive correlated stocks
x = get_top_pos_correlations(df_volume_transposed_pct2, n)
print("Top %d Positive Correlations" % n)
print("For different range of records: ");print(x)

#For top n negative correlated stocks
r = get_top_neg_correlations(df_volume_transposed_pct2, n)   
print("Top %d Negative Correlations" % n)
print("For different range of records: ");print(r)

#===============correlation between all columns using heatmap=======================#
corr = df_volume_transposed_pct2.corr()

ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
    )
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
    );
                

#===============recall original dataset ===========================================#
target_volume_price = pd.DataFrame(volume_price)
target_volume_price = target_volume_price.reset_index()
target_volume_price ['changes'] = target_volume_price['close']-target_volume_price['open']
target_volume_price ['target'] = np.where(target_volume_price['changes'] >0, 'up', 'down')
target_volume_price.set_index('name', inplace=True)
target_volume_price = target_volume_price.loc[['FGV-C63', 'ORION-WA', 'SUMATEC','NETX', 'TRIVE', \
                              'ZELAN','DAYA', 'HSI-C3Z', 'HSI-C5A', 'HSI-C5B','DBHD-WA'],\
                                ['day', 'close', 'open', 'high', 'low', 'volume','dif', 'changes', 'target']]
target_volume_price = target_volume_price.reset_index()
target_volume_price.to_csv('target_volume_price1.csv')


# =============================================================================
# Part 4:
# Normalising, PAA & SAX    
# =============================================================================
df_new = pd.read_csv('target_volume_price.csv',usecols=[i for i in range(8) if i != 0]);print(df_new)
#No. of companies with >60 records
listnew = df_new["name"].unique().tolist()
len(listnew)
print(list_new)
df_red = df_new.set_index(['name','day']).dif.dropna()
print(df_red)

scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)  # Rescale time series
n_paa_segments = 10
n_sax_symbols = 10
n_sax_symbols_avg = 10
n_sax_symbols_slope = 6
for i in listnew:
    records = len(df_red[[i]])
    print("stockname"+str(i))      
    scaleddata = scaler.fit_transform(df_red[[i]])
    #print(scaleddata)      
    paa = PiecewiseAggregateApproximation(n_segments=n_paa_segments)
    paa_dataset_inv = paa.inverse_transform(paa.fit_transform(scaleddata))
    # SAX transform
    sax = SymbolicAggregateApproximation(n_segments=n_paa_segments, alphabet_size_avg=n_sax_symbols)
    sax_dataset_inv = sax.inverse_transform(sax.fit_transform(scaleddata))
    # 1d-SAX transform
    one_d_sax = OneD_SymbolicAggregateApproximation(n_segments=n_paa_segments, alphabet_size_avg=n_sax_symbols_avg,
                                                    alphabet_size_slope=n_sax_symbols_slope)
    one_d_sax_dataset_inv = one_d_sax.inverse_transform(one_d_sax.fit_transform(scaleddata))
    plt.figure()
    # First, raw time series
    plt.subplot(2, 2, 1)  
    plt.plot(scaleddata[0].ravel(), "b-")
    plt.title("Raw time series")
    # Second, PAA
    plt.subplot(2, 2, 2)  
    plt.plot(scaleddata[0].ravel(), "b-", alpha=0.4)
    plt.plot(paa_dataset_inv[0].ravel(), "b-")
    plt.title("PAA")
    #SAX plot
    plt.subplot(2, 2, 3)  # Then SAX
    plt.plot(scaleddata[0].ravel(), "b-", alpha=0.4)
    plt.plot(sax_dataset_inv[0].ravel(), "b-")
    plt.title("SAX, %d symbols" % n_sax_symbols)
    
    plt.subplot(2, 2, 4)  # Finally, 1d-SAX
    plt.plot(scaleddata[0].ravel(), "b-", alpha=0.4)
    plt.plot(one_d_sax_dataset_inv[0].ravel(), "b-")
    plt.title("1d-SAX, %d symbols (%dx%d)" % (n_sax_symbols_avg * n_sax_symbols_slope,
                                              n_sax_symbols_avg,
                                              n_sax_symbols_slope))

    plt.tight_layout()
    plt.suptitle('Stockname: ' + i)
    plt.savefig('normalization.png')
    plt.show()

# =============================================================================
# Part 5:
# Exploring other features   
# =============================================================================

#================Cluster Close Price Vs Volatility==========================#
df_new = groups.filter(lambda x : len(x)>=60)
df_new.to_csv('volume_stocks60.csv')
df_transposed1 = df_new.set_index(['name','day']).close.unstack('name')
df_transposed1.head()

df_return3 = df_transposed1.dropna(how='any',axis=0)
print(df_return3)

returns1 = df_return3.mean()
returns1 = pd.DataFrame(returns1)
returns1.columns = ['ClosePrice']
returns1.head()
returns1['Volatility']= df_return.std()* sqrt(61) #have to use with PCT change
print(returns1)
returns2 = returns1.sort_values('ClosePrice', ascending=False)
returns2.to_csv('closeprice.csv')
returns2.head()

returns3 = returns1.sort_values('Volatility', ascending=False)
returns3.to_csv('volatility-all.csv')
returns3.head()


#=================K Mean Clustering Close Price against Volatility============#
#format the data as a numpy array to feed into the K-Means algorithm
data = np.asarray([np.asarray(returns3['ClosePrice']),np.asarray(returns3['Volatility'])]).T

print(data)
 
X = data
distorsions = []
for k in range(2, 20):
    k_means = KMeans(n_clusters=k)
    k_means.fit(X)
    distorsions.append(k_means.inertia_)
 
fig = plt.figure(figsize=(15, 5))
plt.plot(range(2, 20), distorsions)
plt.grid(True)
plt.title('Elbow curve')
plt.xlabel("Number of cluster")
plt.ylabel("WCSS")
plt.savefig('Volume_Elbow Curve_test.png')


#===============5 clusters=====================#
# computing K-Means with K = 5 (5 clusters)
centroids,_ = kmeans(data,5)
# assign each sample to a cluster
idx,_ = vq(data,centroids)
 
# some plotting using numpy's logical indexing
plt.plot(data[idx==0,0],data[idx==0,1],'ob',
     data[idx==1,0],data[idx==1,1],'oy',
     data[idx==2,0],data[idx==2,1],'or',
     data[idx==3,0],data[idx==3,1],'og',
     data[idx==4,0],data[idx==4,1],'om')
plt.plot(centroids[:,0],centroids[:,1],'sg',markersize=8)
plt.title ("Clusters of Stocks")
plt.xlabel("Mean Close Price")
plt.ylabel("Volatility")
plt.savefig('Cluster_close_volitility, K=5.png')
plt.show()

# =============================================================================
# Part 6
# Crawl extra data to compare with predicted Time series in SAS
# =============================================================================
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests
from bs4 import BeautifulSoup
import string
import time
import matplotlib.pyplot as plt
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.piecewise import PiecewiseAggregateApproximation
from tslearn.piecewise import SymbolicAggregateApproximation, OneD_SymbolicAggregateApproximation


urlTheStar='https://www.thestar.com.my/business/marketwatch/stock-list/?alphabet='
alpha = []
for letter in string.ascii_uppercase:
    alpha.append(letter)     
alpha.append('0-9')
print("!!!  Array of chars")
print(alpha)

stockname = []
for i in alpha:
    print("!!!  Now char "+ i)
    browser = webdriver.Firefox(executable_path='/Users/marinashah/Desktop/Data Mining/geckodriver')
    browser.implicitly_wait(40)
    browser.get(urlTheStar + i)
    WebDriverWait(browser,40).until(EC.visibility_of_element_located((By.ID,'marketwatchtable')))
    innerHTML = browser.find_element_by_id("marketwatchtable").get_attribute("innerHTML")
    soup = BeautifulSoup(innerHTML, 'lxml') 
    links = soup.findAll('a')
    for link in links:
        splitlink = link['href'].split('=')
        stock = splitlink[1]
        stockname.append(stock)
        print(stock)
    browser.close()

dict = {'name':stockname}
df_stockname = pd.DataFrame(dict)
df_stockname.to_csv('stockname.csv')


#using the stockname crawled and saved in csv. Then transform dataframe into list
df1 = pd.read_csv('/Users/marinashah/Desktop/Stock Data/stockname.csv',usecols=[1])
datanames = df1['name'].tolist()

sl=[];cl=[];ol=[];hl=[];ll=[];dl=[];vl=[];stocknames2=[]  

#set timeframe to crawl e.g. 3 months
startdate=str(1554253200) #date = Tuesday, Ja, 2019 7:50:31 PM
enddate=str(1554426000) #date = Tuesday, April 2, 2019 7:50:31 PM 

for name in datanames:
    url = 'https://charts.thestar.com.my/datafeed-udf/history?symbol='+name+'&resolution=D&from='+startdate+'&to='+enddate
    r = requests.get(url).json() 
    if r["s"] == "ok":
        stocknames2.append(name)
        for t in r["t"]:
            day=time.strftime("%Y-%m-%d",time.localtime(int(t)))
            dl.append(day)
            sl.append(name)
        for o in r["o"]:ol.append(o) #open price
        for c in r["c"]:cl.append(c) #closing price
        for h in r["h"]:hl.append(h) #high price
        for l in r["l"]:ll.append(l) #low price
        for v in r["v"]:vl.append(v) #volume
    print("Done for "+ name)
    #break       
    
df = pd.DataFrame({'name':sl,'day':dl,'close':cl,'open':ol,'high':hl,'low':ll,'volume':vl})
df.to_csv('price_df_short.csv')

    

