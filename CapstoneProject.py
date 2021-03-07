#!/usr/bin/env python
# coding: utf-8

# ### IBM Applied Data Science Capstone Project

# # Segmenting and Clustering Neighborhoods in London
# In this notebook, neighborhoods in the city of London are explored, segmented, and clustered. For the London neighborhood data, a Wikipedia page exists that has all the information needed to explore and cluster the neighborhoods in London. The data is scraped from the Wikipedia page and wrangled, cleaned and then read into a pandas dataframe so that it is in a structured format.
# 
# Once the data is in a structured format, Analyze to open a Japanese restaurant and where would we recommend that to open it? 

# ### Our goal is to perform a Segmenting and Clustering Neighborhoods in London and assist  who is looking to open an Japanese restaurant in the city of London with some recommendations.
# 
# ### Introduction 
# 
# ### Business Problem
# 
# The aim is to help restaurant chain owners and/or investors who are looking to open and/or invest in an Japanese restaurant in the city of London.
# 
# To solve this problem, we need to complete the following steps with London data:
# 
# ### List of Neighborhoods in London			
# This London data is extracted from the Wikipedia page titled, ‘List of areas of London’ (https://en.wikipedia.org/wiki/List_of_areas_of_London). Using the BeautifulSoup and Requests packages of Python, the required data is scraped from the webpage. 
# 
# ### Latitude and Longitude coordinates of city London
# We will fetch the location data of from the Python Geocoder package.
# 
# ### Venue data (powered by Foursquare), particularly related to Japanese Restaurants
# In order to use the Foursquare API, we fetch the location data of all these neighbourhoods from the Python Geocoder package. Next, the Foursquare API is used to get the venues of neighborhoods. 
# 
# ### Clustering
# 
# To prepare the data for K-means clustering, we group the data frame by neighborhoods. Lastly, K-means clustering in performed on this data set to return 4 clusters, or categories of neighborhoods in terms of number of Japanese Restaurants.
# 
# This project would be encompassing a series of Data Science techniques, including, Web Scraping (using BeautifulSoup and Requests), Data Cleaning, Data Wrangling and Machine Learning (K-Means clustering algorithm).

# ### Source Of the Data
# https://en.wikipedia.org/wiki/List_of_areas_of_London

# ## Table of Contents
# 
# 1. <a href="#item1">Web-scrape and Explore Dataset</a>
# 
# 2. <a href="#item2">Fetch Latitude and Longitude of each Neighborhood</a>
# 
# 3. <a href="#item3">Explore and Cluster the Neighborhoods in London</a>
# 
# 4. <a href="#item4">Analyze Each Neighborhood</a>
# 
# 5. <a href="#item5">Cluster Neighborhoods</a>
# 
# 6. <a href="#item6">Examine Clusters</a>   
# 
# 7. <a href="#item7">Discussion & Conclusion</a>   
# 

# # Methodology

# Install the required packages.

# In[2]:


#!pip install arcgis
#!pip install wikipedia
#!conda install -c conda-forge geopy --yes
#!pip install geocoder
#!pip install folium
print('Libraries Installed.')


# ### Importing the required packages.

# In[3]:


import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import geocoder # to get coordinates
import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

import folium # map rendering library

import wikipedia as wp

import folium # map rendering library

from arcgis.geocoding import geocode
from arcgis.gis import GIS
gis = GIS()
print('Libraries imported.')


# The task is to explore the city and plot the map to show the Neighborhoods being considered and then build our model by clustering all of the similar Neighborhoods together and finally plot the new map with the clustered Neighborhoods.

# ## 1. Web-scrape and Explore Dataset
# ### Exploring London City

# ### Neighborhoods of London
# 
# Collecting data needed for the our business solution from Wiki.
# 
# ### Data Collection

# In[4]:


#Get the html source
html = wp.page("List of areas of London").html().encode("UTF-8")
df = pd.read_html(html, flavor='html5lib')[1]     
df.head()


# ### Data Preprocessing

# In[5]:


df.rename(columns=lambda x: x.strip().replace(" ", "_"), inplace=True)
df.head()


# ### Feature Selection
# Keep only relavant boroughs, Post town and district for further steps.

# In[6]:


df1 = df.drop( [ df.columns[0], df.columns[4], df.columns[5] ], axis=1)
df1.columns = ['borough','town','post_code']
df1['borough'] = df1['borough'].map(lambda x: x.rstrip(']').rstrip('0123456789').rstrip('['))
df1.head()


# Dimension of the dataframe

# In[7]:


df1.shape


# We currently have 531 records and 3 columns of our data. Lets do the Feature Engineering

# In[8]:


df1 = df1[df1['town'].str.contains('LONDON')]
df1.head()


# In[127]:


df1.shape


# We now have only 308 rows. We can proceed with our further steps. Getting some descriptive statistics

# ## 2. Fetch Latitude and Longitude of each Neighborhood

# We need to get the geographical co-ordinates for the Neighborhoods to plot out map. We have arcgis package for that. 

# In[9]:


def get_x_y_uk(address1):
   lat_coords = 0
   lng_coords = 0
   g = geocode(address='{}, London, England, GBR'.format(address1))[0]
   lng_coords = g['location']['x']
   lat_coords = g['location']['y']
   return str(lat_coords) +","+ str(lng_coords)


# Checking geographical co-ordinates

# In[10]:


CoordinatesUK = df1['post_code']    
CoordinatesUK.head()


# Passing postal codes of london to get the geographical co-ordinates

# In[11]:


LatLngUK = CoordinatesUK.apply(lambda x: get_x_y_uk(x))
LatLngUK.head()


# ### Latitude
# 
# Extracting the latitude from our previously collected coordinates

# In[16]:


LatUK = LatLngUK.apply(lambda x: x.split(',')[0])
LatUK.head()


# ### Longitude
# 
# Extracting the Longitude from our previously collected coordinates

# In[17]:


LngUK = LatLngUK.apply(lambda x: x.split(',')[1])
LngUK.head()


# We now have the geographical co-ordinates of the London Neighborhoods.
# 
# We proceed with Merging our source data with the geographical co-ordinates to make our dataset ready for the next stage

# In[18]:


LondonLatLng = pd.concat([df1,LatUK.astype(float), LngUK.astype(float)], axis=1)
LondonLatLng.columns= ['borough','town','post_code','latitude','longitude']
LondonLatLng.head()


# In[19]:


LondonLatLng.dtypes


# ### Co-ordinates for London
# 
# Getting the geocode for London

# In[20]:


LondonGeoCodes = geocode(address='London, England, GBR')[0]
LondonLat = LondonGeoCodes['location']['y']
LondonLng = LondonGeoCodes['location']['x']
print(LondonLat)
print(LondonLng)


# ## Results : Visualize the Map of London

# In[21]:


# Creating the map of London
LondonMap = folium.Map(location=[LondonLat, LondonLng], zoom_start=12)
LondonMap

# adding markers to map
for latitude, longitude, borough, town in zip(LondonLatLng['latitude'], LondonLatLng['longitude'], LondonLatLng['borough'], LondonLatLng['town']):
    label = '{}, {}'.format(town, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [latitude, longitude],
        radius=5,
        popup=label,
        color='red',
        fill=True
        ).add_to(LondonMap)  
    
LondonMap


# ## 3.Explore and Cluster the Neighborhoods in London
# To proceed with the next part, we need to define Foursquare API credentials.
# 
# Using Foursquare API, we are able to get the venue and venue categories around each Neighborhood in London.

# In[22]:


CLIENT_ID = 'RDKH2NBTC4WBR4GLAY3VE4LC01ECVNZBDWJL2VVUI3IPQARR' # your Foursquare ID
CLIENT_SECRET = 'JNIOHCCN3IML1NKDG0ZQPPC30LHOCZYNM4SSMFTOSYXSVLHY' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version
LIMIT = 100 # A default Foursquare API limit value


# In[23]:


LondonLatLng.head()


# Defining a function to get the neraby venues in the Neighborhood. This will help us get venue categories which is important for our analysis

# In[24]:


LIMIT=100

def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        #print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius,
            LIMIT
            )
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Latitude', 
                  'Longitude', 
                  'Venue', 
                  'Venue Category']
    
    return(nearby_venues)


# Collect the venues in London

# In[25]:


#save the neighbours data ane use it due to foursqure per day limitation
VenuesDf = getNearbyVenues(LondonLatLng['borough'], LondonLatLng['latitude'], LondonLatLng['longitude'])
VenuesDf.head()


# In[26]:


# saving the London Venues dataframe 
#VenuesDf.to_csv('LondonVenues.csv', index=False)
#VenuesDf = VenuesDf.drop(VenuesDf.columns[[0]], axis=1)
#VenuesDf = pd.read_csv("LondonVenues.csv") 
#VenuesDf.head()


# In[27]:


VenuesDf.columns = ['Neighborhood', 'Latitude', 'Longitude', 'VenueName', 'VenueCategory']
VenuesDf.sort_values(["Neighborhood"], inplace=True, ascending=True)


# In[28]:


VenuesDf.shape


# 10354 records for venues.

# ### Grouping by Venue Categories
# 
# Unique Venue Categories

# In[29]:


VenuesDf.groupby(["Neighborhood"]).count()
print('There are {} uniques categories.'.format(len(VenuesDf['VenueCategory'].unique())))


# We can see 298 records.

# ## 4. Analyze Each Neighborhood
# 
# ### One Hot Encoding 
# We need to Encode our venue categories to get a better result for our clustering

# In[30]:


# one hot encoding
onehot = pd.get_dummies(VenuesDf[['VenueCategory']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
onehot['Neighborhoods'] = VenuesDf['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [onehot.columns[-1]] + list(onehot.columns[:-1])
onehot = onehot[fixed_columns]

print(onehot.shape)

grouped = onehot.groupby(["Neighborhoods"]).mean().reset_index()

print(grouped.shape)
grouped.head()


# ### Analysis of Top Restaurant Venues Category in London

# In[31]:


CategoryDf = pd.DataFrame(VenuesDf['VenueCategory'].value_counts()).reset_index()
CategoryDf.columns = ['VenueCategory','Total']
CategoryDf = CategoryDf[CategoryDf['VenueCategory'].str.contains('Restaurant')]
CategoryDf.head()


# ### Bar Chart for Venue Category wise Count

# In[32]:


import plotly.express as px
topvenues_barchart = px.bar(CategoryDf.query("Total>48"),
                            x="VenueCategory",
                            y="Total", 
                            color="VenueCategory")

topvenues_barchart.update_layout(title = 'London Venue Category with venues',
                         margin={"r":0,"t":30,"l":0,"b":0})

topvenues_barchart.update_xaxes(showticklabels=False) # Removed tick labels as it was too long
topvenues_barchart.show() # Display plot


# In[34]:


# len(grouped[grouped["Japanese Restaurant"] > 0])
LondonRest = grouped[["Neighborhoods","Japanese Restaurant"]]
LondonRest


# We can see 50 records, just goes to show how diverse and interesting the place is.

# ### 5. Cluster Neighborhoods

# In[35]:


kclusters = 4

LondonClustering = LondonRest.drop(["Neighborhoods"], 1)
#LondonClustering.head()

# run k-means clustering
kmeans = KMeans(init="k-means++", n_clusters=kclusters, n_init=12).fit(LondonClustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10]


# In[36]:


Merged = LondonRest.copy()

# add clustering labels
Merged["Category"] = kmeans.labels_
Merged.rename(columns={"Neighborhoods": "Neighborhood"}, inplace=True)
Merged.sort_values(["Neighborhood"], inplace=True, ascending=False)
Merged.head()


# Neighbourhood Data Wranging

# In[37]:


Londondf = VenuesDf[VenuesDf.columns[0]] 
Londondf = pd.DataFrame(Londondf).reset_index()
#Eliminate the Duplicatte values with Special chars
Londondf = Londondf.replace('Bexley, Greenwich ', 'Bexley, Greenwich', regex=True)
Londondf['ChrLen'] = Londondf['Neighborhood'].str.len()
Londondf = Londondf.drop_duplicates(subset=['Neighborhood'], keep='first')
Londondf.sort_values(["Neighborhood"], inplace=True, ascending=True)
Londondf = pd.DataFrame(Londondf).reset_index() 

#Remove unwanted columns and indexes 
Londondf = Londondf.drop(Londondf.columns[[0]], axis=1)
Londondf = Londondf.drop(Londondf.columns[[0]], axis=1)
Londondf = Londondf.drop(Londondf.columns[[1]], axis=1)
Londondf.head()


# In[38]:


# Geographical coordinates of neighborhoods
textList = []
neighborhoodList = []
# define a function to get coordinates
def get_latlng(neighborhood):
    # initialize your variable to None
    lat_lng_coords = None
    # loop until you get the coordinates
    while(lat_lng_coords is None):
        g = geocoder.arcgis('{}, London, England, GBR'.format(neighborhood))
        lat_lng_coords = g.latlng
    return lat_lng_coords

coords = [ get_latlng(neighborhood) for neighborhood in Londondf["Neighborhood"].tolist() ]

CoordsDf = pd.DataFrame(coords, columns=['Latitude', 'Longitude'])

# merge the coordinates into the original dataframe
Londondf['Latitude'] = CoordsDf['Latitude']
Londondf['Longitude'] = CoordsDf['Longitude']

# check the neighborhoods and the coordinates
print(Londondf.shape)
Londondf.head()


# ### Adding Neighborhood into the mix.

# In[39]:


MergedDf = Merged.merge(Londondf)
MergedDf.head()

#Sort
MergedDf.sort_values(["Category"], inplace=True, ascending=False)
MergedDf


# ### Results : Visualising Clusters

# In[40]:


# Creating the map of London
ClustersMap = folium.Map(location=[LondonLat, LondonLng], zoom_start=12)
ClustersMap

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i+x+(i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for latitude, longitude, poi, cluster in zip(MergedDf['Latitude'], MergedDf['Longitude'], MergedDf['Neighborhood'], MergedDf['Category']):
    label = folium.Popup(str(poi) + ' - Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [latitude, longitude],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(ClustersMap)  
    
ClustersMap


# ## 6. Examine Clusters

# Category 1: Neighborhoods with very low number of restaurants

# In[41]:


Merged.loc[Merged['Category'] == 0]


# Category 2: Neighborhoods with low number of restaurants

# In[42]:


Merged.loc[Merged['Category'] == 1]


# Category 3: Neighborhoods with a significant number of restaurants

# In[43]:


Merged.loc[Merged['Category'] == 2]


# Category 4: Neighborhoods crowded with restaurants

# In[44]:


Merged.loc[Merged['Category'] == 3]


# ## 7. Discussion & Conclusion

# It is clear the Category 3 are very crowded with Japanese Restaurants, and hence, Category 2 would be the best bet for opening a new restuarant because of not too much competition in these regions, but still a proven market. Client with USPs to stand out from the competition can also open new restaurants in neighborhoods in Cluster 1 with moderate competition.

# # References:
# * [London Areas Wiki](https://en.wikipedia.org/wiki/List_of_areas_of_London)
# * [Foursquare API](https://foursquare.com/)
# * [ArcGIS API](https://www.arcgis.com/index.html)
# 
# <hr>

# ### Thank You
