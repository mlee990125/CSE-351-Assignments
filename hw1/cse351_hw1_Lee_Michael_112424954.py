#!/usr/bin/env python
# coding: utf-8

# In[192]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


# In[111]:


get_ipython().system('pip install wordcloud')


# In[112]:


from wordcloud import WordCloud


# In[2]:


## Q0 Data Exploration
airbnb = pd.read_csv('./airbnb data/AB_NYC_2019.csv')
airbnb


# In[3]:


airbnb.columns


# In[4]:


# can select which columns to remove if unnecessary 
# airbnb_subset = airbnb[['id', 'name', 'host_id', 'host_name', 'neighbourhood_group',
#        'neighbourhood', 'latitude', 'longitude', 'room_type', 'price',
#        'minimum_nights', 'number_of_reviews', 'last_review',
#        'reviews_per_month', 'calculated_host_listings_count',
#        'availability_365']]


# In[5]:


# List the counts of missing values for each column
airbnb.isna().sum()


# In[6]:


# List all the rows with name = NaN
airbnb[airbnb['name'].isna()]


# In[7]:


# Summary of numerical values while dropping useless numerical values(id, host_id, etc.)
num_airbnb = airbnb.drop(['host_id', 'latitude', 'longitude'], axis=1)
num_airbnb.describe()


# In[8]:


# Checking which row had the max price
max_price_row = airbnb.loc[airbnb['price'].idxmax()]
max_price_row


# In[9]:


# Making a histogram of num_airbnb with the bin size and figure size
num_airbnb.hist(bins = 80, figsize=(40,30))


# In[10]:


# Filtering out the rows of price where the price is greather than 2000
high_price = num_airbnb[num_airbnb['price'] > 2000]
high_price


# In[11]:


# Scatter plot of airbnb with high prices
high_price[['id', 'price']].plot(kind="scatter", x="id", y="price", figsize=(20,10))


# In[12]:


# Bar graph of count of neighbourhood names
pd.DataFrame(airbnb['neighbourhood'].value_counts()).plot(kind='bar', figsize=(20, 10))


# In[13]:


# Count of cases where number of reviews = 0
(airbnb['number_of_reviews']==0).sum()


# In[14]:


# Count of each value in each column
for column_name in airbnb.columns:
    print("Value counts for column: ", column_name)
    print(airbnb[column_name].value_counts())
    print('\n')


# In[15]:


# Q1 Data Cleaning
# Check for duplicates
airbnb.duplicated().sum()


# In[16]:


# Check for the missing values
airbnb.isna().sum()


# In[17]:


# Replace all missing values
airbnb['name'].fillna(airbnb['id'], inplace=True)
airbnb['host_name'].fillna(airbnb['host_id'], inplace=True)
airbnb['last_review'].fillna("Not Applicable", inplace=True)
airbnb['reviews_per_month'].fillna(0, inplace=True)


# In[18]:


# Q2 Price vs Neighbourhood
# Filter airbnb so that it has only rows where neighbourhood is listed at least 6 times
airbnb['neighbourhood'].value_counts()
neighbourhood_airbnb = airbnb.groupby('neighbourhood').filter(lambda x: len(x) > 5)
neighbourhood_airbnb


# In[29]:


# Part A
# The top 5 neighbourhoods with the highest prices
# Calculate the means of each neighbourhoods
neighbourhood_prices = neighbourhood_airbnb.groupby('neighbourhood')['price'].mean().reset_index()
neighbourhood_prices


# In[36]:


# Sort neighbourhood_prices
neighbourhood_prices_sorted = neighbourhood_prices.sort_values(by='price', ascending=False)
neighbourhood_prices_sorted


# In[42]:


# Top 5 neighbourhoods with highest prices
neighbourhood_prices_sorted.head(5)


# In[45]:


# Bottom 5 neighbourhoods with the lowest prices
neighbourhood_prices_sorted.tail(5)


# In[46]:


# Part B
# Scatter Plot of price vs neighbourhood group
neighbourhood_airbnb[['neighbourhood_group', 'price']].plot(kind="scatter", x="neighbourhood_group", y="price", figsize=(20,10))


# In[51]:


# Analysis
neighbourhood_airbnb['neighbourhood_group'].value_counts()
# Generally, the more populated the neighbourhood group was, the more expensive it was. 


# In[118]:


# Q3
# I wanted to look at the correlation between the attributes  price, minimum_nights, 
# calculated_host_listings_count, number_of_reviews, and  avability_365.
# Select features for Pearson correlation
pearson_airbnb = airbnb[['price', 'availability_365', 'minimum_nights', 'calculated_host_listings_count', 'number_of_reviews']]
pearson_airbnb


# In[119]:


# Pairwise pearson correlations
correlations = pearson_airbnb.corr()
correlations


# In[178]:


# Heatmap of correlations
sns.heatmap(correlations, annot=True, cmap='coolwarm')


# In[66]:


# Most positive correlation: calculated_host_listings count and availability_365 with a value of 0.23
# Most negative correlation: minimum_nights and number_of_reviews with a value of -0.08


# In[90]:


# Q4
# Part A
locations_airbnb = airbnb[['latitude', 'longitude', 'neighbourhood_group']]
locations_airbnb.loc[:,'neighbourhood_group'] = locations_airbnb['neighbourhood_group'].astype('category')


# In[93]:


# create a dictionary mapping codes to neighborhood names
neighborhoods = dict(enumerate(locations_airbnb['neighbourhood_group'].cat.categories))

# create the scatter plot with color-coded points
ax = locations_airbnb.plot(kind='scatter', x='latitude', y='longitude', figsize=(20, 10), c=locations_airbnb['neighbourhood_group'].cat.codes, colormap='viridis')

# add a color key legend to the plot
colorbar = plt.colorbar(ax.collections[0])
colorbar.set_ticks(range(len(neighborhoods)))
colorbar.set_ticklabels([neighborhoods[key] for key in neighborhoods.keys()])


# In[106]:


# Part B
# Scatter plot with price as color-code
airbnb_location_price = airbnb.loc[airbnb['price'] < 1000, ['latitude', 'longitude', 'price']]

airbnb_location_price.plot(kind="scatter", x='latitude', y='longitude', c='price', figsize=(20, 10), colormap='viridis')


# In[107]:


# We can see the colors of green and dark blue which are more expensive than the majority of colors, (purple)
# are most prevalent in the Manhattan, Brooklyn, and Queens area. The ones with yellow colors are mostly seen
# in the Manhattan area.


# In[114]:


# Q5
# Extract the names of the listings
names = airbnb['name'].astype(str)

# Combine all the names into a single string
all_names = ' '.join(names)

# Generate the word cloud
wordcloud = WordCloud(background_color='white', width=800, height=400).generate(all_names)

# Plot the word cloud
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[174]:


# Q6
# Identify "high" host listing counts
host_counts = airbnb[airbnb['calculated_host_listings_count'] > 10]

# Group the data by neighbourhood group and count the number of unique host IDs
grouped = host_counts.groupby('neighbourhood_group')['host_id'].nunique()

# Plot the results
fig, ax = plt.subplots(figsize=(20, 10))
ax.bar(grouped.index, grouped.values)
ax.set_xlabel('Neighbourhood Group')
ax.set_ylabel('Number of Hosts with High Number of Listings')
ax.set_title('Frequency of Hosts with High Number of Listings per Neighbourhood Group')
plt.show()


# In[177]:


# Calculate the correlation between the number of listings and other factors such as availability and price
corr_matrix = airbnb[['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'availability_365', 'calculated_host_listings_count']].corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(20, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation matrix between factors')
plt.show()


# In[179]:


# Although it isn't a significant correlation, we can see based on the Pearson correlation that 
# the more available the room was, the more host_listings_count it had. 


# In[190]:


# Q7
# Plot 1: I will plot number of reviews per unique id vs calculated host listing count


# In[189]:


# group the data by host_id and calculate the number of reviews per host
reviews_per_host = airbnb.groupby('host_id')['number_of_reviews'].sum()

# group the data by host_id and calculate the calculated_host_listings_count per host
listings_per_host = airbnb.groupby('host_id')['calculated_host_listings_count'].max()

# plot a scatter plot of listings per host versus reviews per host
plt.scatter(listings_per_host, reviews_per_host)
plt.xlabel('Calculated Host Listings Count')
plt.ylabel('Number of Reviews')
plt.show()

# This plot an attempt to show something I believed would be true: that if a host has high number of listings, 
# they would also have high number of reviews. However, it was the opposite. A lot of low listing counts had
# high number of reviews.


# In[193]:


# Plot 2: I will plot a 3d scatter plot of the location of each listings and the price
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

x = airbnb['longitude']
y = airbnb['latitude']
z = airbnb['price']

ax.scatter(x, y, z, c=z, cmap='viridis')

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('Price')

plt.show()

# This plot shows a better visualization of part B of Q4. It shows visually  which areas 
# are more expensive and I believe it has more clarity since the previous one had too much
# overlap in colors. 

