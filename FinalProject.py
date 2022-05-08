
#import spotipy
#from spotipy.oauth2 import SpotifyClientCredentials

#import time
#import shutil
#import re
#from types import MemberDescriptorType
#import discogs_client
#import requests
#import cv2
#import csv

#cid = '16f1343fb87b4e5b9eea7dcff1123613'
#secret = '330ce74bb10e426b841045d8d9ad491b'
#client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
#sp = spotipy.Spotify(client_credentials_manager
#=
#client_credentials_manager)

#artist_name = []
#track_name = []
#popularity = []
#track_id = []

#members = []

#d = discogs_client.Client('ExampleApplication/0.1',user_token = "ZXCxtAiEVIIfDHntXFYLldVILUQhoCzwcXNKpxiw")
##results = d.search(artist='Rammstein')

## release = d.artist(956139)
## print(release.members)
#query = []
#counter = 0
##for i in range(344,345,1):
#for i in range(10000,10090,10):
#    try:
        
#    # artist = d.artist(92476)
#        time.sleep(1)
#        #Throttle queries to 60 per minute max
#        artist_orig = d.artist(i)
#        artist = d.artist(i)
#        inx = []
#        artist = str(artist)

#        #Find max index of last occurence of an int
#        for i in range(0,10,1):
#            indx = artist.rfind(str(i))
#            inx.append(indx)
#        indx = 10 + (len(str(i))+1)
#        artist = artist[indx+1:len(artist)-1]

#        #Fetched Discog album name, now reverse-lookup with Spotify
#        track_results = sp.search(q='artist:'+ artist, type='artist', limit=1)

#        if not not track_results["artists"]["items"]: 
#            artist_name = track_results["artists"]["items"][0]["name"]
#            artist_id = track_results["artists"]["items"][0]['id']

#            #Fetch album list of artist and grab the first one at random
#            albumResults = sp.artist_albums(artist_id, limit=3)
#            albumResults = albumResults['items'][0]

#            #Fetch album URL for lookup of name of album
#            url = albumResults['images'][0]['url']

#            #We assume there's at least one member in the band
#            if len(artist_orig.members) == 0:
#                members.append(1)
#            else: 
#                members.append(len(artist_orig.members))

#            # Use album url to fetch album art and save it 
#            pic = requests.get(url, stream=True)
#            print(members)
#            counter = counter + 1
#            albumname = albumResults['name']
#            with open('img' + str(counter) + '.png', 'wb') as out_file:
#                shutil.copyfileobj(pic.raw, out_file)
#            del pic
#    except:
#        pass

#with open('members.csv', 'w') as f:
      
#    # using csv.writer method from CSV package
#    write = csv.writer(f)
      
#    write.writerow(members)

##print(artist_name)
#    #num_members = (int(len(artist.members)))
    
#    #artist = str(artist).strip('"')
    
#    #print(artist[artist.find('"'):])
#    #query.append([artist,num_members])
#    #print(query)
#    #exit()
# -----------------------------------------------------------------------------#
import cv2
import os
import torch
from IPython.display import Image  # for displaying images
import random
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np 

import streamlit as st
import pandas as pd



st.title("Cover Album Detection via Feature Matching and CNN")

st.write("Here is the dataset used in this analysis:") 

image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])


original_image = Image.open(image_file)
original_image = np.array(original_image)

st.image([original_image])

exit()
#def images_from_path():
#    f = open('train2017_imgs.txt','r')
#    a = f.read()
#    a = a.split("\n")

#    for path in a:
#        if path != '' and os.path.exists(path):
#            load_Image = cv2.imread(path)
#            cv2.imshow('img',load_Image)
#            cv2.waitKey(0)

#def recover_image_name():
#    files = Path('./labels').glob('*.txt')

#    text = list()
#    for file in files:
#        blah = str(file)
#        blah = blah.strip('.txt\/labels')
#        image_nm = str(blah + '.jpg')
    
#    return image_nm

#def extract_info_from_xml():
#    files = Path('./labels').glob('*.txt')

#    text = list()
#    # Initialise the info dict 
#    info_dict = {}
#    info_dict['bboxes'] = []

#    # Parse the XML Tree
#    for file in files:
#        # Get the file name 
#        blah = str(file)
#        blah = blah.strip('.txt\/labels')
#        image_nm = str(blah + '.jpg')

#        info_dict['filename'] = image_nm

#        with file.open() as f:
            
#            # Get the image size
           
#            # Get details of the bounding box 
#            elif elem.tag == "object":
#                bbox = {}
#                for subelem in elem:
#                    if subelem.tag == "name":
#                        bbox["class"] = subelem.text
                    
#                    elif subelem.tag == "bndbox":
#                        for subsubelem in subelem:
#                            bbox[subsubelem.tag] = int(subsubelem.text)            
#                info_dict['bboxes'].append(bbox)
    
#    return info_dict

random.seed(0)


def plot_bounding_box(image, annotation_list):
    annotations = np.array(annotation_list)
    w, h = image.size
    
    plotted_image = ImageDraw.Draw(image)

    transformed_annotations = np.copy(annotations)
    transformed_annotations[:,[1,3]] = annotations[:,[1,3]] * w
    transformed_annotations[:,[2,4]] = annotations[:,[2,4]] * h
    print('are',transformed_annotations)
    transformed_annotations[:,1] = transformed_annotations[:,1] - (transformed_annotations[:,3] /2)
    transformed_annotations[:,2] = transformed_annotations[:,2] - (transformed_annotations[:,4] /2)
    transformed_annotations[:,3] = transformed_annotations[:,1] + transformed_annotations[:,3]
    transformed_annotations[:,4] = transformed_annotations[:,2] + transformed_annotations[:,4]

    for ann in transformed_annotations:
        obj_cls, x0, y0, x1, y1 = ann
        #print(x0, y0, x1, y1)
        #x2_0 = x0 - (x1/2)
        #y2_0 = y0 + (y1/2)
        #x2_1 = x0 + (x1/2)
        #y2_1 = y0 - (y1/2)
        plotted_image.rectangle(((x0,y0), (x1,y1)),width = 5, fill = None,outline = (255,0,0))
        
        plotted_image.text((x0, y0), str(obj_cls))
    
    plt.imshow(np.array(image))
    plt.show()


files = Path('./labels/train').glob('*.txt')

text = list()
# Initialise the info dict 
info_dict = {}
info_dict['bboxes'] = []
counter = 0
# Parse the XML Tree
for file in files:
    with file.open() as f:
        #print(f.read())

        blah = str(file)
        print(blah)
        blah = blah.strip('.txt\/labels/train')
        image_nm = str(blah + '.jpg')
        print(image_nm)
        counter += 1
        annotation_list = f.read().split("\n")[:-1]
        annotation_list = [x.split(" ") for x in annotation_list]
        annotation_list = [[float(y) for y in x ] for x in annotation_list]
        print(annotation_list)
        if counter == 7:
            break
path = 'C:/Users/John E/Downloads/FinalProject/FinalProject/CrowdHuman/Images/'
path = path + image_nm
print(path)
#path = 'C:/Users/John E/Downloads/FinalProject/FinalProject/CrowdHuman/Images/273271,1b86f000bc5b77bf.jpg'
image = Image.open(path)


plot_bounding_box(image,annotation_list)
#load_Image = cv2.imread(path)
#cv2.imshow('img',load_Image)
#cv2.waitKey(0)








#if __name__ == "__main__":

#   # extract_info_from_xml()




