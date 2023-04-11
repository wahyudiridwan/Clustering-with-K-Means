#!/usr/bin/env python
# coding: utf-8

# ### Clustering with K-Means
# 

# by : Wahyudi Arlinawan Ridwan

# #### Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Reading Data

# In[2]:


df = pd.read_csv('Data untuk use case.csv')
df.head()


# #### Cleansing Data

# In[3]:


df = df.set_index(['CustomerID'])  # Make the Customer ID column into an index


# In[ ]:





# In[4]:


Gender = {'Gender':{'Male':1 , 'Female':0}}  # Transforming the Gender column
df.replace(Gender, inplace=True)
df.head()


# ##### Eksploration Data

# In[5]:


df.describe()  # Summary of Numerical Data Statistics Calculation


# In[6]:


df.nunique()  # "Count Distinct"/Unique data checking


# In[7]:


df.isnull().sum()  # Null value data checking


# #### Pair Plotting Data

# In[8]:


sns.pairplot(df)  # Correlation plot between variables


# #### Identifying number of clusters with Elbow Method

# In[9]:


wcss = list()
for i in range(2,10):     # Use number of clusters in range 2-10
    kmeans = KMeans(i)
    kmeans.fit(df)
    wcss.append(kmeans.inertia_)
plt.plot(range(2,10),wcss)


# First hypothesis :The number of clusters obtained based on the elbow method is 5 or 6, so I have to try another way.

# #### Use Silhouette Index to Indetifying Number of Clusters

# In[10]:


from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

sil = list()
for j in range(2,10):   # Use number of clusters in range 2-10
    algorithm = (KMeans(n_clusters = j))
    algorithm.fit(df)
    labels = algorithm.labels_
    sil.append(silhouette_score(df, labels, metric = 'euclidean'))
    
sil
    
    


# Second hypothesis : Based on the silhouette score obtained, it is possible that the cluster number is at number 5 or 6 and is still in line with the results obtained using the elbow method. So we have to prove both numbers

# In[ ]:





# First test with 6

# In[11]:


kmeans = KMeans(6)   
kmeans.fit(df)
clusters = df.copy()
clusters['Cluster'] = kmeans.fit_predict(clusters)
clusters.head(10)


# It turns out that the highest predictive number is 5 resulting in 6 clusters. So that we can try to make an plot using several variables.

# In[ ]:





# In[12]:


# We try to mapping data clustering using 6 clusters

plt.scatter(df['Purchasing power (1-100)'], df['Telco Spending (IDR 000)'], c=clusters['Cluster'], cmap='rainbow')
plt.xlabel('Purchasing power (1-100)')
plt.ylabel('Telco Spending (IDR 000)')
plt.title('Graph of Telco spending and Purchasing power clustering results')
plt.show()


# The results of the mapping are obtained, if using 6 clusters, the results are not in accordance with the clustering requirements, namely homogeneous within the circle and the distance is different from between circles, due to the presence of clusters that accumulate in one circle.

# In[ ]:





# Second Test with 5

# In[13]:


kmeans = KMeans(5)   # Trying use 5 cluster
kmeans.fit(df)
clusters = df.copy()
clusters['Cluster'] = kmeans.fit_predict(clusters)
clusters.head()


# It turns out that the highest cluster number is 4 and resulting in 5 clusters. So that we can try to make an plot using several variables.

# In[14]:


# We proof by mapping the data by grouping into 5 clusters 

plt.scatter(df['Purchasing power (1-100)'], df['Telco Spending (IDR 000)'], c=clusters['Cluster'], cmap='rainbow')
plt.xlabel('Purchasing power (1-100)')
plt.ylabel('Telco Spending (IDR 000)')
plt.title('Graph of Telco spending and Purchasing power clustering results')
plt.show()


# The results are in accordance with the clustering requirements, namely homogeneous within one circle and the distance is different from other circles. So the most optimal cluster is 5 clusters for grouping this data

# In[ ]:





# In[40]:


# The average value of several variables based on the cluster results

mean = clusters.groupby('Cluster').mean()
mean


# In[42]:


# Customers grouping into priority levels based on the results of the average value

mean['Priority Level'] = ['Third Priority','Last Priority','Main Priority','Fourth Priority','Second Priority']
mean


# In[43]:


# Number of Customers per Clusters

count_gender = pd.DataFrame(clusters.groupby(['Cluster'])['Gender'].count())
count_gender


# In[ ]:





# In[ ]:


# Import data transformation and data clustering result into CSV for visualization needs


# In[ ]:


mean.to_csv('Data_01.csv')


# In[44]:


kmeans = KMeans(5)   
kmeans.fit(df)
clusters = df.copy()
clusters['Cluster'] = kmeans.fit_predict(clusters)

#Tranformasi gender
gender = {'Gender label':{1:'Male', 0:'Female'}}
gender_label = {'Gender label':{1:'Male', 0:'Female'}}
clusters['Gender label'] = clusters['Gender']
clusters.replace(gender_label, inplace=True)

# Tranformasi Age ke dalam bentuk age range
bins = [17,26,36,46,56,66,76]
labels = ['17-25', '26-35', '36-45', '46-55', '56-65', '65+']
clusters['Age range'] = pd.cut(clusters.Age, bins, labels = labels, include_lowest = True)
# Labeling age range
bins = [17,26,36,46,56,66,76]
labels = ['Teenegers', 'early adulthood', 'late adulthood', 'early elder', 'late elder', 'aged']
clusters['Age range label'] = pd.cut(clusters['Age'], bins, labels = labels, include_lowest = True)

# Transformasi Telco spending ke dalam bentuk range
bins = [10,36,71,101,151]
labels = ['10-35', '36-70', '71-100', '101+']
clusters['Telco spending range(IDR 000)'] = pd.cut(clusters['Telco Spending (IDR 000)'], bins, labels=labels, include_lowest=True )
# Labeling Telco spending
bins = [10,36,71,101,151]
labels = ['Low', 'Middle', 'High', 'Very high']
clusters['Telco spending range label(IDR 000)'] = pd.cut(clusters['Telco Spending (IDR 000)'], bins, labels=labels, include_lowest=True )

# Transformasi Purchasing power ke dalam bentuk range
bins = [0,26,51,76,101]
labels = ['0-25', '26-50', '51-75', '76+']
clusters['Purchasing power range (1-100)'] = pd.cut(clusters['Purchasing power (1-100)'], bins, labels=labels, include_lowest=True )
# Labeling Purchasing power
bins = [0,26,51,76,101]
labels = ['weak', 'Moderate', 'Strong', 'Very strong']
clusters['Purchasing power range label (1-100)'] = pd.cut(clusters['Purchasing power (1-100)'], bins, labels=labels, include_lowest=True )

clusters


# In[ ]:


clusters.to_csv('Data_01.csv')

