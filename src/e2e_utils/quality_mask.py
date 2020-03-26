from PIL import Image
import numpy as np
import os
from tqdm import tqdm

def get_pixel_array(image_path):
    im = Image.open(image_path)
    im = im.convert("HSV")
    pixels = list(im.getdata()) #Convert to numerical array
    w, h = im.size
    pixels = [pixels[i * w:(i + 1) * w] for i in range(h)] #Stack into matrix
    return pixels, w, h

def apply_mask(image, w, h, threshold_score, center_area=0.1, center_weight=20, border_area=0.3, border_weight=0):
    total_size=w*h
    t_max_count=0
    for y in range(h):
        for x in range(w):
            h, s, v = image[y][x]
            l = (255 - s)/255
            
            if(l > threshold_score):
                t_max_count += 1
                
            
    t_max_count = (t_max_count/total_size)*100
    return t_max_count


imdir = "/home/yeung/workspace/data/normalized/val/"
sex = ["female/","male/"]
scores=[]
threshold = 88
fileGood = open('ok_quality_images','a')
fileBad = open('bad_quality_images','a')

fileGood.write("Origin Folder: "+imdir+"\n Threshold: "+str(threshold)+"\n")
fileBad.write("Origin Folder: "+imdir+"\n Threshold: "+str(threshold)+"\n")

for g in sex:
    for path in tqdm(os.listdir(imdir + g)):
        #print("\t#: {}".format(path))
        im, w, h = get_pixel_array(imdir + g + path)
        m = apply_mask(im, w, h, 0.7, 0.2, 1000, 0.2, 0)
        scores.append((path,m))
        if(m > threshold):
            fileBad.write(g+"/"+path+"\n")
            #os.rename((imdir+path), ("C:/Users/dieck/Desktop/threshold_88/"+path))
        else:
            fileGood.write(g+"/"+path+"\n")
        # TODO: Nothing to do :D 
        
fileGood.close()
fileBad.close()