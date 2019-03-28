import numpy as np
import cv2

def get_corners(img):
#     find continous contours and then its mean
#     so the mean of the intersection will come out tobe the corner
    corners = []
    image, contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        corners.append(np.mean(contours[i].reshape(-1,2), axis=0))
        
    return corners

def filter_triangles(img, gabor_sigma):    
    kernel = np.ones((5,5),np.uint8)
    
    g_kernel1 = cv2.getGaborKernel((31, 31), gabor_sigma, 1*np.pi/4, 2, 0.5, 0, ktype=cv2.CV_32F) #gabor filter
    filtered_img1 = cv2.filter2D(img, cv2.CV_8UC3, g_kernel1) # apply filter
    filtered_img1 = cv2.dilate(filtered_img1,kernel,iterations = 2) # dilate the filtered image to make it continous reduces patches in the filtered image
    filtered_img1 = cv2.erode(filtered_img1,kernel,iterations = 2) # erode to decrease the thickness which gets increased due to dilate
    blur = cv2.GaussianBlur(filtered_img1,(5,5),0)
    ret3,th1 = cv2.threshold(blur,180,255,cv2.THRESH_BINARY) #take threshhold

    g_kernel2 = cv2.getGaborKernel((31, 31), gabor_sigma, 2*np.pi/4, 2, 0.5, 0, ktype=cv2.CV_32F)
    filtered_img2 = cv2.filter2D(img, cv2.CV_8UC3, g_kernel2)
    filtered_img2 = cv2.dilate(filtered_img2,kernel,iterations = 2)
    filtered_img2 = cv2.erode(filtered_img2,kernel,iterations = 2)
    blur = cv2.GaussianBlur(filtered_img2,(5,5),0)
    ret3,th2 = cv2.threshold(blur,180,255,cv2.THRESH_BINARY)

    g_kernel3 = cv2.getGaborKernel((31, 31), gabor_sigma, 3*np.pi/4, 2, 0.5, 0, ktype=cv2.CV_32F)
    filtered_img3 = cv2.filter2D(img, cv2.CV_8UC3, g_kernel3)
    filtered_img3 = cv2.dilate(filtered_img3,kernel,iterations = 2)
    filtered_img3 = cv2.erode(filtered_img3,kernel,iterations = 2)
    blur = cv2.GaussianBlur(filtered_img3,(5,5),0)
    ret3,th3 = cv2.threshold(blur,180,255,cv2.THRESH_BINARY)

    ftc1 = 255*np.uint8(th1*th2) #gives right corner
    ftc2 = 255*np.uint8(th2*th3) #gives left corner
    ftc3 = 255*np.uint8(th3*th1) #gives top corner
    
    return [ftc1, ftc2, ftc3]

def filter_squares(img, gabor_sigma):
    kernel = np.ones((5,5),np.uint8)
    
    g_kernel1 = cv2.getGaborKernel((31, 31), gabor_sigma, 0*np.pi/3, 2, 0.5, 0, ktype=cv2.CV_32F)
    filtered_img1 = cv2.filter2D(img, cv2.CV_8UC3, g_kernel1)
    filtered_img1 = cv2.dilate(filtered_img1,kernel,iterations = 2)
    filtered_img1 = cv2.erode(filtered_img1,kernel,iterations = 2)
    blur = cv2.GaussianBlur(filtered_img1,(5,5),0)
    ret3,th1 = cv2.threshold(blur,180,255,cv2.THRESH_BINARY)

    g_kernel2 = cv2.getGaborKernel((31, 31), gabor_sigma, 2*np.pi/4, 2, 0.5, 0, ktype=cv2.CV_32F)
    filtered_img2 = cv2.filter2D(img, cv2.CV_8UC3, g_kernel2)
    filtered_img2 = cv2.dilate(filtered_img2,kernel,iterations = 2)
    filtered_img2 = cv2.erode(filtered_img2,kernel,iterations = 2)
    blur = cv2.GaussianBlur(filtered_img2,(5,5),0)
    ret3,th2 = cv2.threshold(blur,180,255,cv2.THRESH_BINARY)

    fs = 255*np.uint8(th1*th2) # to get corners
    
    return fs

def get_triangles_centroid(img_mul):
    img_mul_gs = np.uint8(np.mean(img_mul, axis=2))
    ret3,th1 = cv2.threshold(img_mul_gs,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    ft = filter_triangles(th1, 5)
    corners = []
    
    for i in range(3):
        corners.append(np.asarray(get_corners(ft[i])))

    centroid = (corners[0] + corners[1] + corners[2])/3
    
    return centroid

def get_squares_centroid(img_mul):
    img_mul_gs = np.uint8(np.mean(img_mul, axis=2))
    ret3,th1 = cv2.threshold(img_mul_gs,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    fs = filter_squares(th1, 5)
    
    corners = get_corners(fs)
    corners = np.asarray(corners)
    
    centroid_base = []
    for i in range(corners.shape[0]):
        if(i%2 == 1):
            centroid_base.append((corners[i] + corners[i-1])/2)

    centroid_base = np.asarray(centroid_base)

    centroid = []
    start = 0
    end = 1
    i = 0
    while i < (centroid_base.shape[0]):
        if (centroid_base[i][1] - centroid_base[i+1][1])**2 < 25:
            end += 1
            i += 1
        else:
            for k in range(end):
                centroid.append((centroid_base[start + k] + centroid_base[start + k + end])/2)
            i = i + end + 1
            start = i
            end = 1
    
    return np.asarray(centroid)

def detect_triangle(img):
    ft = filter_triangles(img, 2)
    
    for i in range(3):
#         if any of the corner doesn't exist implies not a triagle
        c = np.asarray(np.nonzero(ft[i]))
        if c.shape[1] == 0:
            flag = 0
        else:
            flag = 1
        
    return flag

def detect_square(img):
    fs = filter_squares(img, 2)
    
    corners = get_corners(fs)
    
#     if corners = 4 implies square 
    
    if len(corners) == 4:
        flag = 1
    else:
        flag = 0
        
    return flag

def classify(img):
#     returns 0 for square and 1 for triangle and 2 for anything else
    img_gs = np.uint8(np.mean(img, axis=2))
    ret3,th1 = cv2.threshold(img_gs,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    if detect_square(th1):
        return int(0)
    elif detect_triangle(th1):
        return int(1)
    else:
        return int(2)