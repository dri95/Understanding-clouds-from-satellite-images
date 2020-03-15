import os
cwd = os.getcwd()
os.chdir(r"D:\WORK\DA\understanding_cloud_organization")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import seaborn as sns
import plotly.graph_objs as go
traincsv = pd.read_csv("train.csv")
traincsv.head()


#%%
### Null values
Emptyp = (traincsv.EncodedPixels.isna().sum())
allp = traincsv.EncodedPixels.count()
plt.style.use('ggplot')
x = ['Empty', 'NonEmpty']
y = [Emptyp , allp]
x_pos = [i for i, _ in enumerate(x)]
plt.bar(x_pos, y, color='cyan')
plt.xlabel("Encoded Pixels")
plt.ylabel("Count")
plt.xticks(x_pos, x)

for i, v in enumerate(y):
    plt.text(x_pos[i] - 0.1, v + 0.5, str(v))

plt.show()

#%%
### Donut class distribution

traincsv['Image'] = traincsv['Image_Label'].apply(lambda x: x.split('_')[0])
traincsv['Classlable'] = traincsv['Image_Label'].apply(lambda x: x.split('_')[1])

#traincsv = traincsv[['Image_Lable', 'EncodedPixels','Image','Classlable']]

traincsvnonan = traincsv.dropna()

traincsvnonan.head()
Classlables = ['Fish', 'Flower', 'Gravel', 'Sugar']
nb_Classlabels = []
for item in Classlables:
    nb_class = traincsvnonan['Classlable'].str.count(item)
    nb_Classlabels.append(nb_class[nb_class == 1].count())

pp = traincsvnonan['Classlable'].value_counts()


import matplotlib.pyplot as plt
# Pie chart
labels = ['Fish', 'Flower', 'Gravel', 'Sugar']
sizes = nb_Classlabels
#colors
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
 
fig1, ax1 = plt.subplots()
ax1.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90)
#draw circle
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  
plt.tight_layout()
plt.show()

#%%
### Bar class distribution

plt.style.use('ggplot')

x_pos = [i for i, _ in enumerate(labels)]

plt.bar(x_pos, sizes, color = colors)
plt.xlabel("Cloud Class Lables")
plt.ylabel("Count")
plt.title("Class Distribution")

plt.xticks(x_pos, labels)

for i, v in enumerate(sizes):
    plt.text(x_pos[i] - 0.19, v + 0.5, str(v))

plt.show()

#%%
# count of mask per images
class_counts = traincsvnonan.dropna(subset=['EncodedPixels']).groupby('Image')['Classlable'].nunique()
pc = class_counts.value_counts()
Classc = [ '2', '3','1', '4']
plt.style.use('ggplot')
x_pos = [i for i, _ in enumerate(Classc)]
plt.bar(x_pos, pc, color = '#66b3ff')
plt.xlabel("No.of masks (Classes)")
plt.ylabel("No.of Iamges")
#plt.title("Class Distribution")
plt.xticks(x_pos, Classc)
for i, v in enumerate(pc):
    plt.text(x_pos[i] - 0.19, v + 0.5, str(v))

plt.show()


#%%

train = pd.read_csv("train.csv")

train.head()

train[['Image_ID','Image_Label']] = train.Image_Label.str.split('_', expand=True) 


labelcount = train[['Image_ID', 'Image_Label', 'EncodedPixels']].groupby('Image_ID').apply(lambda x: x.dropna()['Image_Label'].values).reset_index()
labelcount = labelcount.rename(columns = {0: 'labels'})
labelcount['label_counts'] = labelcount['labels'].apply(lambda x: len(x))   

labelcount.head()

import pyfpgrowth as fpg

patterns = fpg.find_frequent_patterns(labelcount['labels'], 2)
patternsdf = pd.DataFrame({'LabelAssociations': list(patterns.keys()), 'Instances': list(patterns.values())})

labels = patternsdf.LabelAssociations
instances = patternsdf.Instances
plt.style.use('ggplot')

x_pos = [i for i, _ in enumerate(labels)]

plt.bar(x_pos, instances, color = '#ff9999')

plt.xticks(x_pos, labels,rotation='vertical',fontsize='small')

for i, v in enumerate(instances):
    plt.text(x_pos[i] - 0.19, v + 0.5, str(v))

plt.show()

ax = patternsdf.plot(x = 'LabelAssociations', y = 'Instances', kind = 'bar', color ='#ff9999')
for i in ax.patches:
    ax.text(i.get_x()-0.2, i.get_height()+.5, str(i.get_height()), fontsize=10, color='black')
plt.show()

###Note: the patternsdf data was copied  to excel and plotted shown in the main Report 