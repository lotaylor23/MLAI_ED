#!/usr/bin/env python
# coding: utf-8

# In[1]:


import geopandas as gpd
gpd.datasets.available
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
get_ipython().run_line_magic('matplotlib', 'inline')
import folium
import requests
import mapclassify
import geodatasets
import heapq
import contextily as ctx
import base64
import html
import aspose.words as aw
from folium import plugins
from branca.colormap import linear
import seaborn as sns
import numpy as np
from folium.features import GeoJsonPopup, GeoJsonTooltip
from folium.features import ClickForLatLng, ClickForMarker, LatLngPopup
from folium import IFrame
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import math
from sklearn import preprocessing
from sklearn import utils
import streamlit as st


# In[2]:


####### Dataset for boundaries
#https://services1.arcgis.com/Ua5sjt3LWTPigjyD/arcgis/rest/services/School_Districts_Current/FeatureServer/0/query?outFields=*&where=1%3D1&f=geojson
#https://www.arcgis.com/apps/mapviewer/index.html?url=https://services1.arcgis.com/Ua5sjt3LWTPigjyD/ArcGIS/rest/services/School_Districts_Current/FeatureServer/0&source=sd
#https://schoolsdata2-tea-texas.opendata.arcgis.com/
#https://schoolsdata2-tea-texas.opendata.arcgis.com/datasets/11533aa5e8e24529aa5ac0c455be6ce6/explore
#https://eric.clst.org/tech/usgeojson/
#https://nces.ed.gov/ccd/schoolmap/#
#https://python-visualization.github.io/folium/modules.html
#https://www.latlong.net/
#https://tea-texas.maps.arcgis.com/apps/webappviewer/index.html?id=51f0c8fa684c4d399d8d182e6edd5d97
#https://python-visualization.github.io/folium/plugins.html


# In[2]:


geojson = gpd.read_file("filepath/Current_Districts_2023.geojson")


# In[3]:


df = pd.read_csv("filepath/t1.csv")
df.head()


# In[5]:


df.describe()


# In[6]:


#histogram 
plt.hist(df['Meets - %'])
plt.xlabel('% of Students')
plt.ylabel('# of Districts')

plt.savefig("filepath/# of Districts-% Students Meets.jpg")
plt.show()


# In[8]:


#scatter
z=df[df["ISD"]=="ALIEF ISD (101903)"]
x=df["ISD"]
y=df[ "Meets - %"]
plt.scatter(x,y,alpha=.55, s=100)
plt.scatter(z["ISD"], z[ "Meets - %"], color="yellow")
ax = plt.gca()
# Comment out to unhide X axis label marks
ax.xaxis.set_tick_params(labelbottom=False)
plt.legend(["Texas Meets", "Alief Meets"], bbox_to_anchor=(1.05, 1.0),loc ="upper left")
plt.ylabel('% of Students')
plt.savefig("filepath/Comparison.jpg")
plt.show()


# In[9]:


#barplot meets by group


# In[10]:


b=df.nlargest(5, 'Meets - %')


# In[11]:


b


# In[12]:


b.plot(x="ISD", y=[ "Did Not Meet - %","Meets - %","Approaches - %","Masters - %"], kind="bar", figsize=(9, 8))
plt.ylabel('% of Students')
plt.xticks(rotation = 45)
plt.legend( bbox_to_anchor=(1.2, 1.0),loc ="upper right")

plt.savefig("filepath/Highest 5 performing districts-Meets.jpg")
plt.show()


# In[13]:


#line current for a specified group 


# In[14]:


b.plot(x="ISD", y=[ "Did Not Meet - %","Meets - %","Approaches - %","Masters - %"], kind="line")
plt.ylabel('% of Students')
plt.xticks(rotation = 90)
plt.savefig("filepath/Highest 5 performing districts-Meets Line.jpg")
plt.show()


# In[15]:


z


# In[22]:


#plt.style.use('seaborn')
z.plot(x="ISD", y=[ "Did Not Meet - %","Meets - %","Approaches - %","Masters - %"], kind="bar", figsize=(9, 8))
plt.ylabel('% of Students')
plt.xticks(rotation = 360)
plt.savefig("filepath/Alief current bar.jpg")
plt.show()


# In[23]:


df2 = pd.read_csv("filepath/5yr basicA.csv")
df2.head()


# In[24]:


plt.scatter(y=df2["STAAR - Reading - Number Tested"],x=df2["Admin"],alpha=.55, s=100)
plt.ylabel('# of Students')
plt.xlabel('Admin')
plt.savefig("filepath/Alief Number Tested Historical.jpg")
plt.show()


# In[25]:


#line historical for a specified group 


# In[26]:


df2.plot(x="Admin", y=[ "STAAR - Reading - Number Tested","STAAR - Reading - Average Scale Score",
                       "STAAR - Reading - Did Not Meet - %","STAAR - Reading - Meets - %",
                       "STAAR - Reading - Approaches - %","STAAR - Reading - Masters - %"], kind="line")
plt.legend( bbox_to_anchor=(1.8, 1.0),loc ="upper right")

plt.savefig("filepath/Alief Multiline Historical.jpg")
plt.show()


# In[27]:


df2.plot(x="Admin", y=["STAAR - Reading - Meets - %"], kind="line")
plt.ylabel('% of Students')
plt.savefig("filepath/5yr basicA2.jpg")

plt.show()


# In[28]:


df2.plot(x="Admin", y=["STAAR - Reading - Did Not Meet - %","STAAR - Reading - Meets - %",
                       "STAAR - Reading - Approaches - %","STAAR - Reading - Masters - %"], kind="line")
plt.legend( bbox_to_anchor=(1.8, 1.0),loc ="upper right")

plt.ylabel('% of Students')
#plt.savefig("filepath/5yr basicA2.jpg")

plt.show()


# In[ ]:


doc = aw.Document()
builder = aw.DocumentBuilder(doc)

#builder.insert_image("filepath/5yr basicA.jpg")
builder.insert_image("filepath/ALLSLRNSP.jpg")
builder.insert_image("filepath/SLR2NSP.jpg")
builder.insert_image("filepath/SLR2.jpg")
builder.insert_image("filepath/SLR1NSP.jpg")
builder.insert_image("filepath/SLR1.jpg")
builder.insert_image("filepath/Texas vs Louisiana Results.jpg")
builder.insert_image("filepath/5yr basicA2.jpg")
builder.insert_image("filepath/Alief Multiline Historical.jpg")
builder.insert_image("filepath/Highest 5 performing districts-Meets Line.jpg")
builder.insert_image("filepath/Highest 5 performing districts-Meets.jpg")
builder.insert_image("filepath/# of Districts-% Students Meets.jpg")
builder.insert_image("filepath/Alief Number Tested Historical.jpg")
builder.insert_image("filepath/Comparison.jpg")
builder.insert_image("filepath/Alief current bar.jpg")


doc.save("filepath/FULL GRAPHS.html")


# In[29]:


data = pd.read_csv("filepath/report (5).csv")
data


# In[30]:


# Check the distribution of the variable with seaborn:

sns.set(style="darkgrid")
sns.histplot(data=data, x="Number Tested")
plt.show()


# In[31]:


colormap = linear.YlGn_09.scale(
   data["Number Tested"].min(),
    data["Number Tested"].max())
nt_dict = data.set_index('Org Id')['Number Tested']

nt_dict#['TX']
m = folium.Map([31.5, -100], zoom_start=5.56)
folium.Choropleth(
    geo_data=geojson,
    name = "geojson",
    style_function = lambda x:style,
    fill_opacity=0.3,
    line_weight=2,
    data=data,
    columns=['Org Id','Number Tested'],
    key_on='feature.properties.DISTRICT_N',
).add_to(m)

folium.GeoJson(geojson,
               zoom_on_click=True,
               name='data'
              ).add_to(m)
folium.LayerControl().add_to(m)
m


# In[32]:


map2 = folium.Map(location=[31.5, -100], zoom_start = 5.56)#, height=300, width=500)

folium.Choropleth(
    geo_data=geojson,
    data=data,
    columns=["Org Id", "Number Tested"],
    key_on="feature.properties.DISTRICT_N",
    fill_color="YlGnBu",#RdYlGn
    fill_opacity=.9,
    line_opacity=0.2,
    smooth_factor=0,
    legend_name="% of students by district - Number Tested",
).add_to(map2)

#map2



style_function = lambda x: {'fillColor': '#ffffff', 
                            'color':'#000000', 
                            'fillOpacity': 0.1, 
                            'weight': 0.1}
highlight_function = lambda x: {'fillColor': '#000000', 
                                'color':'#000000', 
                                'fillOpacity': 0.50, 
                                'weight': 0.1}

Mgl3tx = folium.features.GeoJson(
    geojson,
    style_function=style_function, 
    control=False,
    highlight_function=highlight_function, 
    tooltip=folium.features.GeoJsonTooltip(
        fields=['NAME'],#this uses columns only in map boundary file(json columns)
        aliases=['District: '],
        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
    )
)
map2.add_child(Mgl3tx)
map2.keep_in_front(Mgl3tx)
folium.LayerControl().add_to(map2)
# Add a marker to the map
#folium.Marker(location = [29.682720, -95.593239],color='green').add_to(map2)###alief Marker edit correctly

# Coordinates
coords = pd.DataFrame({'lon': [29.682720,30.079729,29.966249,29.794861,29.576350,29.760427],
                       'lat': [-95.593239,-95.417686,-95.705757,-95.816528,-95.765330,-95.369804]})


# Add several markers to the map
for index, row in coords.iterrows():
  folium.Marker(location = [row["lon"], row["lat"]]).add_to(map2)
map2


# In[4]:


map1 = folium.Map(location=[31.5, -100], zoom_start = 5.56)#, height=300, width=500)
#alief = folium.Html('filepath/5yr basicA.html')
#popup=folium.Popup(alief,max_width=500)

                   
folium.Choropleth(
    geo_data=geojson,
    data=df,
    columns=["Org Id", "Meets - %"],
    key_on="feature.properties.DISTRICT_N",
    fill_color="YlGnBu",#RdYlGn
    fill_opacity=.9,
    line_opacity=0.2,
    smooth_factor=0,
    legend_name="% of students by district - Meets",
).add_to(map1)

#map1


#folium.Marker(location,tooltip=html1, popup=popup,icon=folium.Icon(color='red'')).add_to(feature_group)


#encoded = base64.b64encode(open('filepath/5yr basicA.jpg', 'rb').read())
#html = '<img src="data:image/png;base64,{}">'.format
#iframe = IFrame(html(encoded.decode('UTF-8')), width=400, height=350)
#popup = folium.Popup(iframe, max_width=400)





style_function = lambda x: {'fillColor': '#ffffff', 
                            'color':'#000000', 
                            'fillOpacity': 0.1, 
                            'weight': 0.1}
highlight_function = lambda x: {'fillColor': '#000000', 
                                'color':'#000000', 
                                'fillOpacity': 0.50, 
                                'weight': 0.1}
Mgl3tx = folium.features.GeoJson(
    geojson,
    style_function=style_function, 
    control=False,
    highlight_function=highlight_function, 
    tooltip=folium.features.GeoJsonTooltip(
        fields=['NAME'],#this uses columns only in map boundary file(json columns)
        aliases=['District: '],
        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
    )
)
map1.add_child(Mgl3tx)
map1.keep_in_front(Mgl3tx)
folium.LayerControl().add_to(map1)
# Add a marker to the map
#folium.Marker(location = [29.682720, -95.593239],color='orange').add_to(map1)###alief Marker edit correctly
coords = pd.DataFrame({'lon': [29.682720,30.079729,29.966249,29.794861,29.576350,29.760427],
                       'lat': [-95.593239,-95.417686,-95.705757,-95.816528,-95.765330,-95.369804]})
map1.add_child(folium.ClickForMarker(popup="ISD OF INTEREST"))
# Add several markers to the map
for index, row in coords.iterrows():
  folium.Marker(location = [row["lon"], row["lat"]],tooltip=html,
                #popup=["","","","",""],
                #geo_obj.add_child(folium.Popup(max_width=450).add_child(folium.VegaLite(chart))),
                #popup=popup,
                icon=folium.Icon(color='orange')).add_to(map1)

#row['Location']

#map1
# Circle with a radius of 50km
folium.Circle(
    location = [29.682720, -95.593239],
    radius = 50000,
    color = "red",
    fill = True,
    fill_color = "blue").add_to(map1)


plugins.Geocoder().add_to(map1)
#plugins.MiniMap().add_to(map1)

#folium.Marker(location = [29.682720, -95.593239],
#                #popup=["Alief"],
#                chart=chart_func(s)
#                #popup=popup,
#                icon=folium.Icon(color='orange')).add_to(map1)
map1


# In[34]:


#df = pd.read_csv("C:/Users/username/Desktop/test data.csv")
df3 = pd.read_csv("filepath/RLAgr3LA.csv")
df3.head()


# In[36]:


# Import the pandas library
#import pandas as pd
#import matplotlib.pyplot as plt

# Import the data from the web
#df4 = pd.read_csv("filepath/report (5).csv")#,
                   #dtype={"fips": str})
#df.head()


# In[37]:


#scatter
df = pd.read_csv("filepath/t1v2.csv")
z=df[df["ISD"]=="ALIEF ISD (101903)"]
x=df["ISD"]
y=df[ "Meets"]
w=df["Approaches"]
plt.scatter(x,y,alpha=.55, s=100)
plt.scatter(x,w, color="orange")
plt.scatter(df3["Parish"], df3[ "Mastery"], color="red")
plt.scatter(df3["Parish"], df3[ "Basic"], color="green")

plt.scatter(z["ISD"], z[ "Meets"], color="yellow")
plt.scatter(z["ISD"], z[ "Approaches"], color="purple")

ax = plt.gca()
# Comment out to unhide X axis label marks
ax.xaxis.set_tick_params(labelbottom=False)
plt.legend(["Texas Meets", "Texas Approaches","Louisiana Mastery", "Louisiana Basic"], bbox_to_anchor=(1.05, 1.0),loc ="upper left")

plt.savefig("filepath/Texas vs Louisiana Results.jpg")
plt.show()


# In[38]:


#with open('filepath/gz_2010_us_050_00_20m.json', 'r') as f:
#  data2 = json.load(f)


# In[39]:


#mla on 2 years data( pick a method)


# In[40]:


from sklearn.linear_model import LinearRegression
#X =df2["Admin"]
X =np.array([*range(len(df2["Admin"]))])
Y = df2["STAAR - Reading - Meets - %"]
X =X.reshape(-1, 1)
#X = X.iloc[:, 1].values.reshape(-1, 1) # values converts it into a numpy array
#Y = Y.values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column

#X = outdata.iloc[:, 1].values.reshape(-1, 1)  # values converts it into a numpy array
#Y = outdata.iloc[:, 1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column


linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions
Y
#X
#X[3]
#Y_pred


# In[41]:


Y_pred


# In[42]:


plt.plot(X, Y, '.-',markerfacecolor='yellow', markersize=10)
plt.plot(X, Y_pred, '.--',markerfacecolor='red', markersize=10)
axis_scale = 1.5

plt.xlim(-0.5,  3 * axis_scale)
e=plt.xticks() 
plt.xticks(e[0], labels=["before","2017","2018", "2019",
                       "2021", "2022", "2023"])
# Adjust the annotation formatting as needed
plt.suptitle("Simple Linear Regression")
plt.title('Reading Grade 3 - Meets Grade Level')
plt.ylabel('% of Students')
plt.xlabel('Year of Testing')
for x, y, text in zip(X,Y, Y):
    plt.text(x, y, text,
             horizontalalignment='left', 
             verticalalignment='top',
                  fontsize=12)
for x, y, text in zip(X,Y_pred, Y_pred):
    plt.text(x, y, text,
             horizontalalignment='left', 
             verticalalignment='bottom',
                  fontsize=12)
plt.legend(["Actual Values", "Expected Values"], loc ="upper left")

plt.savefig("filepath/SLR1.jpg")
plt.show()


# In[43]:


c=np.array([5])
c=c.reshape(-1,1)
d=linear_regressor.predict(c)

print(d)


# In[44]:


plt.plot(X, Y, '.-',markerfacecolor='yellow', markersize=10)
plt.plot(X, Y_pred, '.--',markerfacecolor='red', markersize=10)
plt.plot(c, d, '.',markersize=10)


axis_scale = 2

plt.xlim(-0.5,  3 * axis_scale)
e=plt.xticks() 
plt.xticks(e[0], labels=["before","2017","2018", "2019",
                       "2021", "2022", "2023","after"])
# Adjust the annotation formatting as needed
plt.suptitle("Simple Linear Regression With Estimated Next Score Prediction")
plt.title('Reading Grade 3 - Meets Grade Level')
plt.ylabel('% of Students')
plt.xlabel('Year of Testing')
for x, y, text in zip(X,Y, Y):
    plt.text(x, y, text,
             horizontalalignment='left', 
             verticalalignment='top',
                  fontsize=12)
for x, y, text in zip(X,Y_pred, Y_pred):
    plt.text(x, y, text,
             horizontalalignment='left', 
             verticalalignment='bottom',
                  fontsize=12)
for x, y, text in zip(c,d, d):
    plt.text(x, y, text,
             horizontalalignment='right', 
             verticalalignment='bottom',
                  fontsize=12)
for x, y, text in zip(c,d, d):
    plt.text(x, y, "ENSP",
             horizontalalignment='center', 
             verticalalignment='top',
                  fontsize=12)    
plt.legend(["Actual Values", "Expected Values", "Est. Next Score Prediction"], loc ="upper left")


plt.savefig("filepath/SLR1NSP.jpg")
plt.show()


# In[45]:


#fit the model
model = LogisticRegression()
model.fit(X, Y)
#the matrix of probabilities that the predicted output is equal to zero or one
model.predict_proba(X)
#the actual predictions, based on the probability matrix and the values of 𝑝(𝑥)
model.predict(X)


# In[46]:


#plots

#model
plt.plot(X,Y, '.-',markerfacecolor='yellow', markersize=10)
plt.plot(X,model.predict(X), '.--',markerfacecolor='red', markersize=10)

plt.xlim(-0.5,  3 * axis_scale)
e=plt.xticks() 
plt.xticks(e[0], labels=["before","2017","2018", "2019",
                       "2021", "2022", "2023","after"])
# Adjust the annotation formatting as needed
plt.suptitle("Simple Logistic Regression")
plt.title('Reading Grade 3 - Meets Grade Level')
plt.ylabel('% of Students')
plt.xlabel('Year of Testing')
for x, y, text in zip(X,Y, Y):
    plt.text(x, y, text,
             horizontalalignment='left', 
             verticalalignment='top',
                  fontsize=12)
for x, y, text in zip(X,model.predict(X), model.predict(X)):
    plt.text(x, y, text,
             horizontalalignment='right', 
             verticalalignment='bottom',
                  fontsize=12)
plt.legend(["Actual Values", "Expected Values"], loc ="upper left")


plt.savefig("filepath/SLR2.jpg")
plt.show()


# In[47]:


f=np.array([5])
f=f.reshape(-1,1)
g=model.predict(f)

print(g)


# In[48]:


#plots

#model
plt.plot(X,Y, '.-',markerfacecolor='yellow', markersize=10)
plt.plot(X,model.predict(X), '.--',markerfacecolor='red', markersize=10)
plt.plot(f, g, '.', markersize=10)
plt.xlim(-0.5,  3 * axis_scale)
e=plt.xticks() 
plt.xticks(e[0], labels=["before","2017","2018", "2019",
                       "2021", "2022", "2023","after"])
# Adjust the annotation formatting as needed
plt.suptitle("Simple Logistic Regression With Estimated Next Score Prediction")
plt.title('Reading Grade 3 - Meets Grade Level')
plt.ylabel('% of Students')
plt.xlabel('Year of Testing')
for x, y, text in zip(X,Y, Y):
    plt.text(x, y, text,
             horizontalalignment='center', 
             verticalalignment='bottom',
                  fontsize=12)
for x, y, text in zip(X,model.predict(X), model.predict(X)):
    plt.text(x, y, text,
             horizontalalignment='center', 
             verticalalignment='bottom',
                  fontsize=12)
for x, y, text in zip(f,g, g):
    plt.text(x, y, text,
             horizontalalignment='center', 
             verticalalignment='bottom',
                  fontsize=12)
for x, y, text in zip(f,g, g):
    plt.text(x, y, "ENSP",
             horizontalalignment='center', 
             verticalalignment='top',
                  fontsize=12)    
plt.legend(["Actual Values", "Expected Values", "Est. Next Score Prediction"], loc ="upper left")


plt.savefig("filepath/SLR2NSP.jpg")
plt.show()


# In[49]:


plt.suptitle("Your next % of students Meets GL is predicted to be in 2023: " +"\n")
plt.title("Using Linear Regression: "+str(d[-1]), loc='left',color='grey', style='italic')
plt.title("Using Logistic Regression: "+str(g[-1]), loc='right',color='grey', style='italic')
plt.axis('off')


plt.savefig("filepath/ALLSLRNSP.jpg")
plt.show()


# In[6]:


#can you do this with demographic data? yes I can!!!

df4 = pd.read_csv("filepath/5yr ethnic.csv")
df4.head(5)


# In[7]:


k=df4[(df4["Group"]=="Hispanic/Latino") & (df4["Admin"]=="Spring 2019")]
k.head(5)


# In[8]:


map3 = folium.Map(location=[31.5, -100], zoom_start = 5.56)

folium.Choropleth(
    geo_data=geojson,
    data=k,
    columns=["Org Id", "STAAR - Reading - Meets - %"],
    key_on="feature.properties.DISTRICT_N",
    fill_color="YlGnBu",#RdYlGn
    fill_opacity=.9,
    line_opacity=0.2,
    smooth_factor=0,
    legend_name="% of students by district - Meets",
).add_to(map3)

style_function = lambda x: {'fillColor': '#ffffff', 
                            'color':'#000000', 
                            'fillOpacity': 0.1, 
                            'weight': 0.1}
highlight_function = lambda x: {'fillColor': '#000000', 
                                'color':'#000000', 
                                'fillOpacity': 0.50, 
                                'weight': 0.1}

Mgl3tx = folium.features.GeoJson(
    geojson,
    style_function=style_function, 
    control=False,
    highlight_function=highlight_function, 
    tooltip=folium.features.GeoJsonTooltip(
        fields=['NAME'],
        aliases=['District: '],
        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
    )
)
map3.add_child(Mgl3tx)
map3.keep_in_front(Mgl3tx)
folium.LayerControl().add_to(map3)

coords = pd.DataFrame({'lon': [29.682720],
                       'lat': [-95.593239]})


for index, row in coords.iterrows():
  folium.Marker(location = [row["lon"], row["lat"]],tooltip=html,
                icon=folium.Icon(color='orange')).add_to(map3)

folium.Circle(
    location = [29.682720, -95.593239],
    radius = 50000,
    color = "red",
    fill = True,
    fill_color = "blue").add_to(map3)


plugins.Geocoder().add_to(map3)

map3


# In[ ]:




