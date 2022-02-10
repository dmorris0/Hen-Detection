import numpy as np
import cv2
import pandas as pd
from matplotlib import pyplot as plt

# Should be run from the folder containing label-data csv file

df = pd.read_csv("CollectedData_CZ.csv", header=[0,1,2,3])

def draw_ind(im, dfi):
    """Draw an individual hen"""
    n_points = len(dfi.index.get_level_values(0))
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in np.linspace(0, 1, n_points)]

    lastCent = None
    for c, v in zip(colors, dfi.index.get_level_values(0)):
        gg = dfi.loc[v].T
        if np.isnan(gg['x']) or np.isnan(gg['y']):
            continue
        cent = (int(gg['x']),int(gg['y']))
        c = tuple((x*255 for x in c))
            
        cv2.circle(im,cent,6,c[:3],-1) 
        if v=='blade':
            cv2.circle(im,cent,10,c[:3],2) 

        
        if not lastCent:
            lastCent = cent
        else:
            cv2.line(im, lastCent, cent,[0,255,255],2)
            lastCent = cent


def draw_pic(filename, dfi):
    """Draw all hens in a picture. `dfi` is the row in dataframe for this image"""
    im = cv2.imread(filename)
    for hen in set(list(dfi.index.get_level_values(0))):
        draw_ind(im, dfi[hen])
    
    cv2.imwrite(f"{filename}_out.png", im)


#Draw keypoints in all pics of current folder
for ind, row in df.iterrows():
    file_name = row.to_dict()[('scorer',
                              'individuals',
                              'bodyparts',
                              'coords')]
    file_name = file_name.split('/')[-1]
    draw_pic(file_name, row['CZ'])
