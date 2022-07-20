#!/usr/bin/env python
# coding: utf-8

# # Creating Customer Segments
# 
# ### Unsupervised Learning

# In[1]:


# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import renders as rs
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# #### **Task 1: Import Dataset and create a copy of that dataset**

# In[2]:


# Write code here
data = pd.read_csv("customers.csv")
df = data.copy()


# **Task 2: Drop Region and Channel column**

# In[3]:


# Write code here
df=df.drop(['Region','Channel'], axis=1)


# **Task 3: Display first five rows** 

# In[4]:


# Write code here
df.head()


# #### **Task 4: Display last five rows** 

# In[5]:


# Write code here
df.tail()


# #### **Task 5: Check the number of rows and columns**

# In[6]:


# Write code here
df.shape


# #### **Task 6: Check data types of all columns**

# In[7]:


# Write code here
df.dtypes


# **Task 7: Check for missing values and fill missing values if required.**

# In[8]:


# Write code here
df.isnull().sum()


# ## Data Exploration

# #### **Task 8: Check summary statistics and store the resultant DataFrame in a new variable named *stats***

# In[9]:


# Write code here
stats = df.describe()
stats


# ### Implementation: Selecting Samples
# To get a better understanding of the customers and how their data will transform through the analysis, it would be best to select a few sample data points and explore them in more detail. In the code block below, add **three** indices of your choice to the `indices` list which will represent the customers to track. It is suggested to try different sets of samples until you obtain customers that vary significantly from one another.

# **Logic in selecting the 3 samples: Quartiles**
# - As you can previously (in the object "stats"), we've the data showing the first and third quartiles.
# - We can filter samples that are starkly different based on the quartiles.
#     - This way we've two establishments that belong in the first and third quartiles respectively in, for example, the Frozen category.

# **Task 9: Select any random sample and assign the list to given variable**

# In[10]:


# Write code here
indices = [28,92,289]


# These samples will be separated into another dataframe for finding out the details the type of customer each of the selected respresents

# **Task 10: Make a dataframe of selected indices**

# In[11]:


# Write code here
samples = pd.DataFrame(data.loc[indices],columns=data.keys()).reset_index(drop = True)
samples


# The selected sample values should be ranked amongst the whole of the data values to check their ranks and get a better understanding of spending of each sample/customer in each category

# In[13]:


percentiles = df.rank(pct=True)
percentiles = 100*percentiles.round(decimals=3)
percentiles = percentiles.iloc[indices]
percentiles


# **Task 11: Draw a heatmap to show the above results achieved in** `percentile` **to have a better understanding.**

# In[14]:


#Write code here
sns.heatmap(percentiles,annot=True);


# **Task 12: Find the corelation among all the variables of whole dataframe and describe the findings you infer from the heatmapt.**

# In[15]:


# Write the code here
correlation=df.corr()
correlation


# There is a high correlation betwen grocery and detergents_paper however the grocery has negative relation ship with frozen and detergents_paper with fresh.

# ### Pair Plot

# Pairplot is a plot which is used to give and over view of the data in a graphical grid form. The result it shows gives us a picture of variables themselves in a graphical way as well as a relationship of one variable with all the others. For more details you can [click here](https://seaborn.pydata.org/generated/seaborn.pairplot.html)

# **Task 13: Make a pairplot using seaborn.**

# In[16]:


# write code here
sns.pairplot(df);


# ## Data Preprocessing
# In this section, you will preprocess the data to create a better representation of customers by normalizing it by **removing skewness** and **detecting (and optionally removing) outliers**. 

# ### Implementation: Feature Scaling
# If data is not normally distributed, especially if the mean and median vary significantly (indicating a large skew), it is most [often appropriate](http://econbrowser.com/archives/2014/02/use-of-logarithms-in-economics) to apply a non-linear scaling — particularly for financial data.

# **Task 14: Apply log on data for transforming it from skewed to normalized form. Use function** `np.log()` **and save the result in** `log_data`

# In[17]:


#Write code here
log_data = np.log(df)


# ### Implementation: Outlier Detection
# Detecting outliers in the data is extremely important in the data preprocessing step of any analysis. The presence of outliers can often skew results which take into consideration these data points. There are many "rules of thumb" for what constitutes an outlier in a dataset. Here, we will use [Tukey's Method for identfying outliers](http://datapigtechnologies.com/blog/index.php/highlighting-outliers-in-your-data-with-the-tukey-method/): An *outlier step* is calculated as 1.5 times the interquartile range (IQR). A data point with a feature that is beyond an outlier step outside of the IQR for that feature is considered abnormal.
# 
# In the code block below, you will need to implement the following:
#  - Assign the value of the 25th percentile for the given feature to Q1. Use `np.percentile` for this.
#  - Assign the value of the 75th percentile for the given feature to Q3. Again, use `np.percentile`.
#  - Assign the calculation of an IQR for the given feature.
#  - Query the data to filter out Outliers using IQR
#  - remove data points from the dataset by adding indices to the outliers list
# 
# **NOTE:** If you choose to remove any outliers, ensure that the sample data does not contain any of these points! 
# 
# Once you have performed this implementation, the dataset will be stored in the variable `good_data`.

# In[18]:


outliers=[]
# For each feature find the data points with extreme high or low values
for feature in log_data.keys():
    
    # TODO: Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(log_data[feature],25)
    
    # TODO: Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(log_data[feature],75)
    
    # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = 1.5 * (Q3-Q1) 
    
    # Display the outliers
    print("Data points considered outliers for the feature '{}':".format(feature))
    out=log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))]
    display(out)
    outliers=outliers+list(out.index.values)
    
# Select the indices for data points you wish to remove
outliers = list(set([x for x in outliers if outliers.count(x) > 1]))    
print ("Outliers: {}".format(outliers))

# Remove the outliers, if any were specified
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)


# **Task 15: Make a pairplot to check changes in data after pre-processing and using the** `good_data`

# In[19]:


# Write the code here
sns.pairplot(good_data);


# ## Feature Transformation
# In this section you will use principal component analysis (PCA) to draw conclusions about the underlying structure of the wholesale customer data. Since using PCA on a dataset calculates the dimensions which best maximize variance, we will find which compound combinations of features best describe customers.

# ### Implementation: PCA
# 
# Now that the data has been scaled to a more normal distribution and has had any necessary outliers removed, we can now apply PCA to the `good_data` to discover which dimensions about the data best maximize the variance of features involved. In addition to finding these dimensions, PCA will also report the *explained variance ratio* of each dimension — how much variance within the data is explained by that dimension alone. Note that a component (dimension) from PCA can be considered a new "feature" of the space, however it is a composition of the original features present in the data.
# 
# In the code block below, you will need to implement the following:
#  - Import `sklearn.decomposition.PCA` and 
#  - Apply a PCA transformation of the good data.

# **Task 16: Import PCA Library**

# In[20]:


# Write your code here
from sklearn.decomposition import PCA


# **Task 17: Apply PCA by fitting the good data with the same number of dimensions as features.**

# In[21]:


# Write your code here
pca_ = PCA(n_components=good_data.shape[1])


# In[22]:


# Write your code here
pca_.fit(good_data)


# In[23]:


# Generate PCA results plot
pca_results = rs.pca_results(good_data, pca_)
pca_results


# **Task 18: Find cumulative explained variance**

# In[24]:


# Write the code here
cumsum_pca_results= print(np.cumsum(pca_.explained_variance_))


# **Question**
# How much variance in the data is explained ***in total*** by the first and second principal component? What about the first four principal components? How many components should be selected for reducing the dimensions? Give your answer along with the reason.

# **Answer:** the 45 % and 72% variance is explained by first and second component. The two components must be select for reducing dimensions.

# ### Implementation: Dimensionality Reduction
# In the code block below, you will need to implement the following:
#  - Assign the results of fitting PCA in two dimensions with `good_data` to `pca`.
#  - Apply a PCA transformation of `good_data` using `pca.transform`, and assign the results to `reduced_data`.
#  - Apply a PCA transformation of the sample log-data `log_samples` using `pca.transform`, and assign the results to `pca_samples`.

# **Task 19: Apply PCA by fitting the good data with the selected number of components**

# In[25]:


# write your code here
pca = PCA(n_components=2).fit(good_data)


# In[26]:


pca


# **Task 20: Transform the good data using the PCA fit above**

# In[27]:


# write your code here
reduced_data = pca.transform(good_data)


# **Task 21: Create a DataFrame for the reduced data**

# In[28]:


# write your code here
reduced_data =pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])


# ## Implementation: Creating Clusters

# In this section, you will choose to use either a K-Means clustering algorithm  and hierarchical clustering to identify the various customer segments hidden in the data. You will then recover specific data points from the clusters to understand their significance by transforming them back into their original dimension and scale. 

# ## Choosing K

# **Before Implementing KMeans and hierarchical clustering, choose the optimal K using the following method**

# - Silhouette Score

# ### Silhouette Score for K-Means

# In[29]:


# Import necessary libraries
from sklearn.metrics import silhouette_score


# **Task 22: Check Silhouette Score for finding Optimal K**

# In[30]:


# Import necessary libraries
from sklearn.cluster import KMeans


# In[31]:


# write your code here
s_score = []
for k in range(2, 15):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(reduced_data)
    s_score.append([k, silhouette_score(reduced_data, kmeans.labels_)])


# In[32]:


s_score


# **Task 23: Plot a graph representing the Silhouette Score.**

# In[33]:


#add plot
plt.figure(figsize=(15,6))
sns.set_context('poster')
plt.plot( pd.DataFrame(s_score)[0], pd.DataFrame(s_score)[1])
plt.xlabel('clusters')
plt.ylabel('score')
plt.title('The silhouette score') 
plt.show()


# ### Silhouette Score for Hierarchical Clustering

# In[34]:


# Import necessary libraries
from sklearn.cluster import AgglomerativeClustering


# **Task 24: Write the code below to calculate silhouette score for each hierarchical clustering**

# In[35]:


# write the code below
hs_score = []
for k in range(2, 15):
    ward = AgglomerativeClustering(n_clusters=k).fit(reduced_data)
    hs_score.append([k, silhouette_score(reduced_data, ward.fit_predict(reduced_data))])


# In[36]:


hs_score


# **Task 25: Write the code below to make a plot for silhouette score**

# In[37]:


# add plot
plt.figure(figsize=(15,6))
sns.set_context('poster')
plt.plot( pd.DataFrame(hs_score)[0], pd.DataFrame(hs_score)[1])
plt.xlabel('clusters')
plt.ylabel('score')
plt.title('The silhouette score') 
plt.show()


# **Answer:** The 2 number of clusters have best silhouette score.

# ## Implementation of K-Means

# **Task 26: Implement KMeans using your choosen K**

# In[38]:


# write your code here
kmean = KMeans(n_clusters=2, random_state=0)
kmean.fit(reduced_data)


# In[39]:


# write your code here
pred=reduced_data.copy()
pred['kmean'] = kmean.labels_
pred['kmean'].value_counts()


# In[53]:


pred.head()


# ## Implementation Hierarchical Clustering

# **Task 27: Implement Hierarchical(agglomerative) clustering using your choosen K**

# In[40]:


# write your code here
a_cluster = AgglomerativeClustering(n_clusters=2)


# In[41]:


# write your code here
preds_agg = a_cluster.fit_predict(reduced_data)


# ## Best Clustering Algorithm?

# **You will be using** `adjusted rand index` **to select the best clustering algorithm by comparing each of the calculated labels with actual labels found in** `data['Channel]` . Before calculating the score, we need to make sure that the shape of true labels is consistent with the resultant labels.

# In[42]:


true_labels = data['Channel'].drop(data['Channel'].index[outliers]).reset_index(drop = True)


# In[43]:


true_labels.shape


# **Task 28: Find the adjusted rand index for K-Means and Agglomerative Clustering**

# In[44]:


# Import necessary libraries
from sklearn.metrics.cluster import adjusted_rand_score


# In[45]:


kmeans_score = adjusted_rand_score(true_labels, kmean.labels_)
print ('The score for Kmeans is ',kmeans_score)


# In[46]:


aggl_score = adjusted_rand_score(true_labels, preds_agg)
print ('The score for Agglomerative Clustering is ', aggl_score)


# **Question: Which has the best score and should be selected?**

# **Answer:**The k means has a good score and should be selected.

# ## Visualizing the clusters

# **Task 29: Get the centers for KMeans**

# In[47]:


# Write code here
centers =  kmeans.cluster_centers_
centers


# In[48]:


rs.cluster_results(reduced_data, pred, centers)


# # Profiling

# In[57]:


df_pred = df.drop(df.index[outliers]).reset_index(drop = True)
df_pred['pred'] = kmean.labels_


# In[58]:


df_pred


# **Task 30: Get the average prices for each category from the original data frame for each cluster and then make a profile for each**

# In[61]:


# write the code here
clustered_avg = df_pred.groupby('pred')["Fresh", "Milk","Grocery","Frozen","Detergents_Paper", "Delicatessen"].mean().reset_index()
clustered_avg                             


# **Task 31: Make a radar chart to show a better profile for each cluster.**

# In[71]:


# Write the code to import the library files for plotly and set your credentials
import chart_studio
chart_studio.tools.set_credentials_file(username='Amna28', api_key='ywU2GiDvRRsKKtIPnbxb')


# In[72]:


# write the code here
import chart_studio.plotly as py
import plotly.graph_objs as go


# **Task 32: Make the data set for radar chart**

# In[73]:


# Write your code here
radar_data = radar_data = [
    go.Scatterpolar(
      r = list(clustered_avg .loc[0,[ "Fresh", "Milk","Grocery","Frozen","Detergents_Paper", "Delicatessen",'Fresh']]),
      theta = ["Fresh", "Milk","Grocery","Frozen","Detergents_Paper", "Delicatessen",'Fresh'],
      fill = None,
      fillcolor=None,
      name = 'Cluster 0'
    ),
    go.Scatterpolar(
      r = list(clustered_avg .loc[1,["Fresh", "Milk","Grocery","Frozen","Detergents_Paper", "Delicatessen",'Fresh']]),
      theta = ["Fresh", "Milk","Grocery","Frozen","Detergents_Paper", "Delicatessen",'Fresh'],
      fill = None,
      fillcolor=None,
      name = 'Cluster 1'
    )
    
]


# **Task 33: Set the layout for your radar chart and plot it**

# In[74]:


# Write your code here
radar_layout = go.Layout(polar = dict(radialaxis = dict(visible = True,range = [0, 9000])), showlegend = True)


# In[75]:


# add plot
fig = go.Figure(data=radar_data, layout=radar_layout)
py.iplot(fig, filename = "radar")


# ## Conclusion

# In this final section, you will investigate ways that you can make use of the clustered data. First, you will consider how the different groups of customers, the ***customer segments***, may be affected differently by a specific delivery scheme. Next, you will consider how giving a label to each customer (which *segment* that customer belongs to) can provide for additional features about the customer data. Finally, you will compare the ***customer segments*** to a hidden variable present in the data, to see whether the clustering identified certain relationships.

# ### Visualizing Underlying Distributions
# 
# At the beginning of this project, it was discussed that the `'Channel'` and `'Region'` features would be excluded from the dataset so that the customer product categories were emphasized in the analysis. By reintroducing the `'Channel'` feature to the dataset, an interesting structure emerges when considering the same PCA dimensionality reduction applied earlier to the original dataset.
# 
# Run the code block below to see how each data point is labeled either `'HoReCa'` (Hotel/Restaurant/Cafe) or `'Retail'` the reduced space. In addition, you will find the sample points are circled in the plot, which will identify their labeling.

# In[76]:


# Display the clustering results based on 'Channel' data
rs.channel_results(reduced_data, outliers)

