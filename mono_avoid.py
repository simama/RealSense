# Monocular Obstacle Avoidance

import cv2
#cv2.ocl.setUseOpenCL(False)
import scipy.ndimage as ndi
from scipy.stats import itemfreq
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from skimage.feature import local_binary_pattern
from matplotlib import pyplot as plt
from itertools import product
import lbp_stride
import sys

#print cv2.__version__

#def decode_fourcc(v):
#    v = int(v)
#    return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])

# Set video capture parameters: resolution, fps...
def set_param(width=320, height=240):
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print('Initial video resolution: {} x {}'
        .format(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
    return (fps, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# use k-means algorithm to classify color pixels in defined number of buckets
def color_quantization_sk(image, clusters):
    # load the image and grab its width and height
    (h, w) = image.shape[:2]
     
    # convert the image from the RGB color space to the L*a*b*
    # color space -- since we will be clustering using k-means
    # which is based on the euclidean distance, we'll use the
    # L*a*b* color space where the euclidean distance implies
    # perceptual meaning
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
     
    # reshape the image into a feature vector so that k-means
    # can be applied
    image = image.reshape((image.shape[0] * image.shape[1], 3))
     
    # apply k-means using the specified number of clusters and
    # then create the quantized image based on the predictions
    clt = MiniBatchKMeans(n_clusters = clusters)
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]
     
    # reshape the feature vectors to images
    quant = quant.reshape((h, w, 3))
     
    # convert from L*a*b* to RGB
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    return quant

# use k-means algorithm to classify color pixels in defined number of buckets
def color_quantization_cv(image, clusters):
    Z = img.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = clusters
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    return res2

def auto_cany(image, sigma=0.33):
    """http://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-
    edge-detection-with-python-and-opencv/"""

    # compute the median of the single channel pixel intensitres
    image = cv2.bilateralFilter(image,12,20,20) #(11,17,17)
    v = np.median(image)

    # apply automatic Canny edge detection
    lower = int(max(0, (1.0 - sigma)*v))
    upper = int(min(255, (1.0 + sigma)*v))
    edged = cv2.Canny(image,lower, upper)
    #edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    edged = cv2.dilate(edged,kernel,iterations=3)
    cv2.imshow("Edge", edged)
    return edged

def get_contours(image):
    """http://docs.opencv.org/3.1.0/d4/d73/
    tutorial_py_contours_begin.html#gsc.tab=0"""
    img, contours, hierarchy = cv2.findContours(image,cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)
    return img, contours

def hsv_map():
    hsv_map = np.zeros((180, 256, 3), np.uint8)
    h, s = np.indices(hsv_map.shape[:2])
    hsv_map[:,:,0] = h
    hsv_map[:,:,1] = s
    hsv_map[:,:,2] = 255
    hsv_map = cv2.cvtColor(hsv_map, cv2.COLOR_HSV2BGR)
    #cv2.imshow('hsv_map', hsv_map)
    return hsv_map

def set_scale(val):
    global hist_scale
    hist_scale = val

def orb_f(img):
    # Initialize the ORB detector
    orb = cv2.ORB_create()
    # Find the keypoints with ORB
    kp = orb.detect(img,None)
    # Compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    # try:
    #     print kp[0].pt
    # except:
    #     pass    
    img2 = cv2.drawKeypoints(img,kp,dummy,color=(0,255,0), flags=0)
    return kp, des

def sift():
    MIN_MATCH_COUNT = 10

    print ("Position the board and press SPACE to continue")
    k = 0xFF & cv2.waitKey(1)
    while k != ord(' '):
        status, img1 = cap.read()
        k = 0xFF & cv2.waitKey(1)
    status, img1 = cap.read()
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    print ('Please move the patern and press SPACE when you are ready,'
            'or press ESC to cancel the calibration')
    k = 0
    while k != ord(' '):
        status, img2 = cap.read()
        k = 0xFF & cv2.waitKey(1)
        if k == 27:
            return [], []
    status, img2 = cap.read()
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initiate SIFT detector
    sift = cv2.SIFT()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

    return img3, M

def find_features():
    print ("Position the board and press SPACE to continue")
    k = 0xFF & cv2.waitKey(1)
    while k != ord(' '):
        status, img1 = cap.read()
        k = 0xFF & cv2.waitKey(1)
    status, img1 = cap.read()
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    #img1 = img1[80:160, 80:240]
    cv2.imshow("Video", img1)
    kp1, des1 = orb_f(img1)
    img = cv2.drawKeypoints(img1,kp1,dummy,color=(0,255,0), flags=0)
    cv2.imshow("Video", img)
    print ('Please move the patern and press SPACE when you are ready,'
            'or press ESC to cancel the calibration')
    k = 0
    while k != ord(' '):
        status, img2 = cap.read()
        k = 0xFF & cv2.waitKey(1)
        if k == 27:
            return [], []
    status, img2 = cap.read()
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Video", img2)
    kp2, des2 = orb_f(img2) 
    img = cv2.drawKeypoints(img2,kp2,dummy,color=(0,255,0), flags=0)
    cv2.imshow("Video", img)
    print ("Press SPACE to continue")
    k = 0xFF & cv2.waitKey(0)
    print ("Done!")
    return kp1, des1, img1, kp2, des2, img2

def find_homography_matrix():
    kp1, des1, img1, kp2, des2, img2 = find_features()
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # Store all good matches as per Lowe's ratio test
    good = []
    for match in matches:
        if len(match) == 2:
            m,n = match
            if m.distance < 0.7 * n.distance:
                good.append(m)
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        print M
        h,w = img1.shape
        pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, M)

        img2 = cv2.polylines(img2,[np.int32(dst)],True,(0,0,255),3,cv2.LINE_AA)
    else:
        print ("Not enough matches are found - %d/%d") %(len(good),MIN_MATCH_COUNT)
        matchesMask = None

    draw_params = dict(matchColor = (0,0,255),
                        singlePointColor = None,
                        matchesMask = matchesMask,
                        flags = 2)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2,good,None,**draw_params)
    return img3, M

def ground_features(img1):
    kp1, des1 = orb_f(img1)
    print "Move the camera and press SPACE to mark the second frame"
    while k != ord(' '):
        status, img = cap.read()
        k = 0xFF & cv2.waitKey(1)
        if k == 27:
            return
    status, img2 = cap.read()
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Video", img2)
    kp2, des2 = orb_f(img2) 
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    else:
        print ("Not enough matches are found - %d/%d") %(len(good),MIN_MATCH_COUNT)
        matchesMask = None

# calculate hcv color histograms
def calc_hist(hsv,mask, figure=None, ax1=None, ax2=None):
    chans = cv2.split(hsv)
    # Or maybe I should use Numpy indexing for faster splitting: h=hsv[:,:,0] 
    hist_h = cv2.calcHist([chans[0]], [0], mask, [180], [0, 180])
    hist_s = cv2.calcHist([chans[1]], [0], mask, [256], [0, 256])
    #print hist_s
    hist_h = hist_h.flatten()
    hist_s = hist_s.flatten()
    # Apply Gaussian low pass for histogram (using scipy)
    hist_hg = ndi.gaussian_filter1d(hist_h, sigma=1.5, output=np.float64, mode='nearest')
    hist_sg = ndi.gaussian_filter1d(hist_s, sigma=1.5, output=np.float64, mode='nearest')
    hue_max = np.argmax(hist_hg)
    saturation_max = np.argmax(hist_sg) if np.argmax(hist_sg) >= 20 else 20
    #print hue_max, saturation_max
    #ax1.clear(), ax2.clear()
    #ax1.set_autoscale_on(False)
    # ax1.plot(range(180),hist_hg)
    # ax2.plot(range(256),hist_sg)
    # ax1.set_ylim([0,1200])
    # ax1.set_xlim([0,180])
    # ax2.set_xlim([0,256])
    # ax2.set_ylim([0,1200])
    # figure.canvas.draw()
    #plt.xlim([0, 180])
    lower = np.array([hue_max+20,saturation_max-20,20])
    upper = np.array([hue_max+20,saturation_max+20,255])
    mask_color = cv2.inRange(hsv, lower, upper)
    return hue_max, hist_hg, saturation_max, hist_sg, mask_color

def obstacle(img, height, width, hsv, sat):
    for pos in product(range(height),range(width)):
        pixel = img[pos]
        if hsv[pixel[0]] < 60 or sat[pixel[1]] < 80:
            img[pos] = 255
        else:
            img[pos] = 0
    #print img
    return img

def shi_corners(img):
    corners = cv2.goodFeaturesToTrack(img,10,0.01,10)
    corners = np.int0(corners)
    for i in corners:
        print i
        x, y = i.ravel()
        cv2.circle(img,(x,y),3,255,-1)
    plt.imshow(img)
    plt.show() 

def kullback_leibler_divergence(p, q):
    #p = np.asarray(p)
    #q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    #print filt.shape
    #print filt
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))

def corner_hmg(src,dst):
    pass

def nothing(x):
    pass

# this works pretty good
def color_back_projection(hsv, mask, clahe):
    # calculate object histogram
    #hsv = cv2.GaussianBlur(hsv,(3,3),0) # play with this parameter
    #hue = hsv[:,:,0]
    #saturation = hsv[:,:,1]
    #value = hsv[:,:,2]
    #equ1 = cv2.equalizeHist(hue)
    #equ1 = clahe.apply(hue)
    #hsv = cv2.merge((equ1,saturation,value))
    roihist = cv2.calcHist([hsv], [0,1], mask, [180,256], [0,180,0,256])
    
    # normalize histogram and apply backprojection
    cv2.normalize(roihist, roihist,0,255,cv2.NORM_MINMAX)
    dst = cv2.calcBackProject([hsv],[0,1],roihist,[0,180,0,256],1)

    el = cv2.getTrackbarPos('Elipse','Backproject2')
    if el % 2 == 0:
        el += 1
    th = cv2.getTrackbarPos('Threshold','Backproject2')
    ke = cv2.getTrackbarPos('Kernel','Backproject2')
    if ke % 2 == 0:
        ke += 1
    it = cv2.getTrackbarPos('Iterations','Backproject2')

    # kernel for the dilation and erosion
    kernel = np.ones((ke,ke),np.uint8)

    # convolute with circular disc
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(el,el))
    cv2.filter2D(dst,-1,disc,dst)

    # treshold and binary AND
    ret, thresh = cv2.threshold(dst,th,255,0)
    thresh = cv2.merge((thresh,thresh,thresh))
    #thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.dilate(thresh,kernel,iterations=it)
    res = cv2.bitwise_and(hsv,thresh)
    res = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    cv2.imshow("Backproject1", thresh)
    cv2.imshow("Backproject2", res)

# use stride to extract blocs from a matrix
def block_view(A, block= (3, 3)):
    """Provide a 2D block view to 2D array. No error checking made.
    Therefore meaningful (as implemented) only for blocks strictly
    compatible with the shape of A."""
    # simple shape and strides computations may seem at first strange
    # unless one is able to recognize the 'tuple additions' involved ;-)
    shape= (A.shape[0]/ block[0], A.shape[1]/ block[1])+ block
    strides= (block[0]* A.strides[0], block[1]* A.strides[1])+ A.strides
    return ast(A, shape= shape, strides= strides)

# calculate local binary patern histogram
def lbp_histogram(gray, radius, no_points):
    lbp = local_binary_pattern(gray, no_points, radius, method='uniform')
    hist = itemfreq(lbp.ravel())
    hist = hist[:, 1]/sum(hist[:, 1])
    return hist

# use this one, seems faster
def lbp_histogram2(gray, radius, no_points):
    lbp = local_binary_pattern(gray, no_points, radius, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(),
            bins=np.arange(0, no_points + 3),
            range=(0, no_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    #print hist.size
    return hist

# kinda works, but looks like some kind of edge detector
def lbp_compare(gray, radius, no_points):
    template_hist = lbp_histogram2(gray[220:240, 150:170], radius, no_points)
    image = local_binary_pattern(gray, no_points, radius, method='uniform')
    image = np.uint8(image)
    image_comp = template_hist[image]

    # convolute with circular disc
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    cv2.filter2D(image_comp,-1,disc,image_comp)
    #print np.amax(image_comp)
    # treshold and binary AND
    ret, thresh = cv2.threshold(image_comp,5,255,0)
    
    thresh = cv2.merge((thresh,thresh,thresh))
    #print thresh
    #thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.dilate(thresh,kernel,iterations=2)
    #res = cv2.bitwise_and(img,thresh)
    #res = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    cv2.imshow("VBP_Compare", image_comp)
    #cv2.imshow("Backproject2", res)

def lbp_strided(gray, radius, no_points):
    template_hist = lbp_histogram2(gray[200:240, 140:180], radius, no_points)
    for x in xrange(0, 240, 10):
        for y in xrange(0, 320, 10):
            view = gray[x:x+10, y:y+10]
            block_hist = lbp_histogram2(view, radius, no_points)
            #d = cv2.compareHist(np.array(block_hist, dtype=np.float32), np.array(template_hist, dtype=np.float32), cv2.HISTCMP_CHISQR_ALT)
            d = kullback_leibler_divergence(block_hist,template_hist)
            #print d
            if d < 1:
                gray[x:x+10, y:y+10] = 255
            else:
                gray[x:x+10, y:y+10] = 0
    cv2.imshow("Backproject1", gray)

# matching template based        
def template_matching(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(gray)
    img_gray = cl1
    template = cl1[180:240, 130:190]
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.2
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

    #cv2.imshow('res.png',img)


font = cv2.FONT_HERSHEY_SIMPLEX
color = (0, 255, 0)

cap = cv2.VideoCapture(2)

cv2.namedWindow("Video")
cv2.namedWindow("Backproject2")
cv2.createTrackbar('Elipse','Backproject2',5,11,nothing)
cv2.createTrackbar('Threshold','Backproject2',50,255,nothing)
cv2.createTrackbar('Kernel','Backproject2',3,11,nothing)
cv2.createTrackbar('Iterations','Backproject2',2,5,nothing)

fps, width, height = set_param()
status, img = cap.read()
#cv2.createTrackbar("Gaussian", "Video", blur, 15, lambda v: cap.set(cv2.CAP_PROP_FPS, v))
#cv2.createTrackbar("Focus", "Video", focus, 100, lambda v: cap.set(cv2.CAP_PROP_FOCUS, v / 100))

"""
# initialize the histogram
hsv_map = hsv_map()
cv2.namedWindow('hist', 0)
hist_scale = 10
cv2.createTrackbar('scale', 'hist', hist_scale, 32, set_scale)
"""

# draw a sampling polygon for floor recognition
color_polygon = np.array([[width/2-50,height-30],[width/2-50,height],
    [width/2+50,height],[width/2+50,height-30]], np.int32)
color_polygon = color_polygon.reshape((-1,1,2))

calibration_area = np.array([[width/2-80,height-160],[width/2-80,height-80],
    [width/2+80,height-80],[width/2+80,height-160]], np.int32)
calibration_area = calibration_area.reshape((-1,1,2))

corner_area = np.array([[width/2-80,height-130],[width/2-80,height-20],
    [width/2+80,height-20],[width/2+80,height-130]], np.int32)
corner_area = corner_area.reshape((-1,1,2))

# Mask for histogram calculation
mask = np.zeros(img.shape[:2], np.uint8)
mask[height-40:height, width/2-40:width/2+40] = 255
#plt.imshow(mask, cmap = 'gray')
#plt.show()

# kernel for the dilation and erosion
kernel = np.ones((5,5),np.uint8)

# parameters for LBP
radius = 3
no_points = 8 * radius

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# Initialize canvas for histogram plots
# plt.ion()
# fig = plt.figure()
# ax1 = fig.add_subplot(121)
# ax2 = fig.add_subplot(122)
# lines1, = ax1.plot([],[],'-r')
# lines2, = ax2.plot([],[],'-b')

# Dummy tamplate for feature drawing
dummy = np.zeros((1,1))

# This should be LshIndexParams based on miniflann.cpp
FLANN_INDEX_LSH = 6
# FLANN matcher parameters. Commented values are recommended as per docs
# example is using other values since they worked better for them
index_params = dict(algorithm = FLANN_INDEX_LSH,
                    table_number = 12, # 12
                    key_size = 20, # 20
                    multi_probe = 2) # 2
search_params = dict(checks=50)   # or pass empty dictionary
# Number of matches necessary to be considered as a same object for FLANN
MIN_MATCH_COUNT = 6
match_flag = False
while True:
    status, img = cap.read()
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img,(5,5),0)
    #img = color_quantization_sk(img, 16)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

##### find floor based on color histogram (cv2.calcBackProject)
    #color_back_projection(hsv,mask,clahe)

    # find floor texture and match (Local Binary Pattern)
    #lbp_compare(img_gray, radius, no_points)
    #lbp_strided(img_gray, radius, no_points)
    
    # for use with cython code; no significant improvement 
    #img_gray = np.int32(img_gray)
    #imgs = lbp_stride.lbp_strided(img_gray, radius, no_points)
    #imgs = np.asarray(imgs)
    #cv2.imshow("Backproject1", imgs)

    # Calculate color histograms and return dominant hue and saturation
    #hue, hist_h, saturation, hist_s, mask_color = calc_hist(hsv,mask)
    #m = obstacle(hsv, height, width, hist_h, hist_s)
    
##### calculate edges in the picture
    #img_gray2 = auto_cany(img_gray)
    #img2, contours = get_contours(img_gray2)
    #cv2.drawContours(img, contours, -1, (255,255,0), thickness=-1)
    
    # template matching based on grid search and similarity 
    #template_matching(img_gray)

    #img2 = img
    
    # Calculate features and dispay them
    #kp, des = orb_f(img_gray)

    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sample_region = img[height-20:height, width/2-30:width/2+30]
    
 

    # calculates the color histogram
    #small = cv2.pyrDown(hsv)
    dark = sample_region[...,2] < 32
    sample_region[dark] = 0
    h = cv2.calcHist([sample_region], [0, 1], None, [180, 256], [0, 180, 0, 256])
    h = np.clip(h*0.005*hist_scale, 0, 1)
    vis = hsv_map*h[:,:,np.newaxis] / 255.0
    cv2.imshow('hist', vis)
    #hue_sum = np.sum(h, axis = 1)
    #floor_color = np.argmax(hue_sum)
    #print np.amax(hue_sum)
    #print floor_color
    cluster_img = color_quantization_cv(sample_region, 1)
    print cluster_img[0][0]
    floor_color = cv2.cvtColor(np.reshape(cluster_img[0][0],(1,1,3)), cv2.COLOR_BGR2HSV)
    lower_bound = np.uint8([[[floor_color[0][0][0]-10,50,50]]])
    upper_bound = np.uint8([[[floor_color[0][0][0]+10,255,255]]])

    # Threshold the HSV image to get only target colors
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img,img, mask= mask)



    
    #img = color_quantization_sk(img, 8)
    """
    
    #cv2.putText(img, "Resolution: {0} x {1}".format(width,height), (15, 20), 
    #    font, 0.30, color)
    #cv2.putText(img, "FPS: {}".format(fps), (15, 40), font, 0.30, color)
    
    

    # Draw only keypoints locations, not size and orientation
    #img2 = cv2.drawKeypoints(img_gray,kp,dummy,color=(0,255,0), flags=0)
    

    #res = cv2.bitwise_and(img2,img2, mask= mask_color)
    #cv2.polylines(img,[calibration_area],True,(0,0,255))
    cv2.imshow("Video", img)
    #cv2.imshow("Obstacle", m)
    #cv2.imshow("Polygon", cv2.cvtColor(sample_region, cv2.COLOR_HSV2BGR))
    
    #cv2.imshow("Polygon", sample_region)
    #cv2.imshow("Mask", mask_color)
    #cv2.imshow("Result", res)
    #cv2.imshow("Cluster", cluster_img)
    

    k = 0xFF & cv2.waitKey(1)

    if k == 27:
        break
    elif k == ord('c'):
        img_calib, M = find_homography_matrix()
        #img_calib, M = sift()
        plt.imshow(img_calib, 'gray')
        plt.show()
    elif k == ord('t'):
        shi_corners(img_gray[110:220, 80:240])
        #ground_features(img2)

cap.release()
cv2.destroyAllWindows()
