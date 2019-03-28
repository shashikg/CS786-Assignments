import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io
from numpy.random import choice as randI
from numpy.random import permutation

img_tri = cv2.imread('triangle.jpg',0)
img_sq = cv2.imread('square.jpg',0)

# Search space is a 1440 x 1440 image which is divided into 144 blocks of
# 120 x 120 matrixes

def feature_search_paradigm(N):
    img = np.zeros((1440,1440,3), np.uint8)
    color = [0, 2]
    object_location = permutation(np.arange(144))[:N] #randomly take points for object location out of avaliable 144 position
    odd_stimuli_location = int(randI(N,1))
    
    odd_stimuli_color = int(color[int(randI(2,1))])
    like_stimuli_color = int(2 - odd_stimuli_color)
    
#     matrix for location k in the block of 144 in 1440 x 1440 image will be
#     img[(k/12)*120:(k/12+1)*120, (k%12)*120:(k%12+1)*120,:]
    for i in range(N):
        if i == odd_stimuli_location:
            img[int(object_location[i]/12)*120:(int(object_location[i]/12)+1)*120, int(object_location[i]%12)*120:(int(object_location[i]%12)+1)*120, odd_stimuli_color] = img_sq
        else:
            img[int(object_location[i]/12)*120:(int(object_location[i]/12)+1)*120, int(object_location[i]%12)*120:(int(object_location[i]%12)+1)*120, like_stimuli_color] = img_sq

    return (object_location, img)

def conjunction_search_paradigm(N):
    color = [0, 2]
    img = np.zeros((1440,1440,3), np.uint8)
    N_of_sq = int(randI(N-1,1)) + 1 #randomly takeno  of squares
    N_of_tr = N - N_of_sq
    
    object_location = permutation(np.arange(144))[:N] #randomly take points for object location out of avaliable 144 position
    square_location = object_location[:N_of_sq] #randomly take points for square object out of avaliable N object
    triangle_location = object_location[N_of_sq:]
    
    tr_color = int(color[int(randI(2,1))])
    sq_color = int(2 - tr_color)
    
    odd_stimuli_shape = int(randI(2,1))
    
#     matrix for location k in the block of 144 in 1440 x 1440 image will be
#     img[(k/12)*120:(k/12+1)*120, (k%12)*120:(k%12+1)*120,:]
    if odd_stimuli_shape:
        odd_stimuli_location = int(randI(N_of_sq,1))
        
        for i in range(N_of_sq):
            if i == odd_stimuli_location:
                img[int(square_location[i]/12)*120:(int(square_location[i]/12)+1)*120, int(square_location[i]%12)*120:(int(square_location[i]%12)+1)*120, tr_color] = img_sq
            else:
                img[int(square_location[i]/12)*120:(int(square_location[i]/12)+1)*120, int(square_location[i]%12)*120:(int(square_location[i]%12)+1)*120, sq_color] = img_sq
                
        for i in range(N_of_tr):
                img[int(triangle_location[i]/12)*120:(int(triangle_location[i]/12)+1)*120, int(triangle_location[i]%12)*120:(int(triangle_location[i]%12)+1)*120, tr_color] = img_tri
    else:
        odd_stimuli_location = int(randI(N_of_tr,1))
        
        for i in range(N_of_tr):
            if i == odd_stimuli_location:
                img[int(triangle_location[i]/12)*120:(int(triangle_location[i]/12)+1)*120, int(triangle_location[i]%12)*120:(int(triangle_location[i]%12)+1)*120, sq_color] = img_tri
            else:
                img[int(triangle_location[i]/12)*120:(int(triangle_location[i]/12)+1)*120, int(triangle_location[i]%12)*120:(int(triangle_location[i]%12)+1)*120, tr_color] = img_tri
                
        for i in range(N_of_sq):
                img[int(square_location[i]/12)*120:(int(square_location[i]/12)+1)*120, int(square_location[i]%12)*120:(int(square_location[i]%12)+1)*120, sq_color] = img_sq


    return (object_location, img)

def get_paradigm(N, E):
    if E == 0:
        object_location, paradigm = feature_search_paradigm(N)
    else:
        object_location, paradigm = conjunction_search_paradigm(N)
        
    return (object_location, np.uint8(paradigm))