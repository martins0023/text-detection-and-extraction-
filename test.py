#importing of modules to be used to run the program
import extract
import csv
import os
import csv
import cv2
import logging
import pytesseract
import pandas as pd
import numpy as np
from scipy.stats import mode
from PIL import Image
import argparse
import os
import random


#from google.colab.patches import cv2_imshow

#import detectron2
#from detectron2.utils.logger import setup_logger
#setup_logger()
#from detectron2 import model_zoo
#from detectron2.engine import DefaultPredictor
#from detectron2.config import get_cfg
#from detectron2.utils.visualizer import Visualizer

import logging

import cv2
import numpy as np
from scipy.stats import mode

#import and unzip the dataset
#!ls
#!unzip "Text_Detection_Dataset_COCO_Format.zip"


#preparing the imported and extracted dataset with json

#import json
#from detectron2.structures import BoxMode
#def get_board_dicts(imgdir):
#    json_file = imgdir+"/dataset.json"
#    with open(json_file) as f:
#        dataset_dicts = json.load(f)
#    for i in dataset_dicts:
#        filename = i["file_name"] 
#        i["file_name"] = imgdir+"/"+filename 
#        for j in i["annotations"]:
#            j["bbox_mode"] = BoxMode.XYWH_ABS
#            j["category_id"] = int(j["category_id"])
#    return dataset_dicts


#preprocessing the image pre-processing and pattern matching.

#This python module can perform the following functions:

#Binarization - method binary_img(img) performs this function
#Skew correction - method skew_correction(img) performs this function
#Need to introduce machine learning of some sort to make the skew correction method run faster :( Or... A simple fix would be to resize the #image first, and then apply the skew correction method! That'll probably take lesser time...




logging.basicConfig(
  level=logging.DEBUG,
  format="%(levelname)s: %(asctime)s {%(filename)s:%(lineno)d}: %(message)s "
)

kernel = np.ones((5, 5), np.uint8)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img = cv2.imread('image_resize.jpg') # read image file to be processed

"""
Method to binarize an image
Input: Grayscale image
Output: Binary image
The nature of the output is such that the text(foreground) has a colour 
value of (255,255,255), and the background has a value of (0,0,0).
"""


def binary_img(img):
  # img_erode = cv2.dilate(img,kernel,iterations = 2)
  blur = cv2.medianBlur(img, 5)

  # mask1 = np.ones(img.shape[:2],np.uint8)
  """Applying histogram equalization"""
  cl1 = clahe.apply(blur)

  circles_mask = cv2.dilate(cl1, kernel, iterations=1)
  circles_mask = (255 - circles_mask)

  thresh = 1
  circles_mask = cv2.threshold(circles_mask, thresh, 255, cv2.THRESH_BINARY)[1]

  edges = cv2.Canny(cl1, 100, 200)

  edges = cv2.bitwise_and(edges, edges, mask=circles_mask)

  dilation = cv2.dilate(edges, kernel, iterations=1)

  display = cv2.bitwise_and(img, img, mask=dilation)

  cl2 = clahe.apply(display)
  cl2 = clahe.apply(cl2)

  ret, th = cv2.threshold(cl2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  th = 255 - th

  thg = cv2.adaptiveThreshold(display, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                              cv2.THRESH_BINARY, 11, 2)

  # final = cv2.bitwise_and(dilation,dilation,mask=th)

  finalg = cv2.bitwise_and(dilation, dilation, mask=thg)

  finalg = 255 - finalg

  abso = cv2.bitwise_and(dilation, dilation, mask=finalg)

  return abso


"""
Method to resize the image. This is going to help in reducing the number 
of computations, as the size of data will reduce.
"""


def resize(img):
  r = 1000.0 / img.shape[1]
  dim = (1000, int(img.shape[0] * r))
  resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

  # cv2.imshow('resized', resized)
  return resized


"""
Method to correct the skew of an image
Input: Binary image
Output: Skew corrected binary image
The nature of the output is such that the binary image is rotated appropriately
to remove any angular skew.
Find out the right place to insert the resizing method call.
Try to find one bounding rectangle around all the contours
"""


def skew_correction(img):
  areas = []  # stores all the areas of corresponding contours
  dev_areas = []  # stores all the areas of the contours within 1st std deviation in terms of area#stores all the white pixels of the largest contour within 1st std deviation
  all_angles = []
  k = 0

  binary = binary_img(img)
  # binary = resize(binary)
  im2, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  # cnt = contours[0]
  # upper_bound=len(contours)
  height_orig, width_orig = img.shape[:2]
  words = np.zeros(img.shape[:2], np.uint8)

  for c in contours:
    areas.append(cv2.contourArea(c))

  std_dev = np.std(areas)
  for i in areas:
    dev_areas.append(i - std_dev)

  dev_contours = np.zeros(img.shape[:2], np.uint8)

  for i in dev_areas:
    if ((i > (-std_dev)) and (i <= (std_dev))):
      cv2.drawContours(dev_contours, contours, k, (255, 255, 255), -1)
    k += 1

  sobely = cv2.Sobel(dev_contours, cv2.CV_64F, 0, 1, ksize=5)
  abs_sobel64f = np.absolute(sobely)
  sobel_8u = np.uint8(abs_sobel64f)

  cv2.imshow('Output2',sobel_8u)

  minLineLength = 100
  maxLineGap = 10
  lines = cv2.HoughLinesP(sobel_8u, 1, np.pi / 180, 100, minLineLength, maxLineGap)

  for x1, y1, x2, y2 in lines[0]:
    cv2.line(words, (x1, y1), (x2, y2), (255, 255, 255), 2)
  # cv2.imshow('hough',words)

  height_orig, width_orig = img.shape[:2]
  all_angles = []

  im2, contours, hierarchy = cv2.findContours(words, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  logging.debug(len(contours))
  contour_count = 0
  for c in contours:
    # max_index = np.argmax(areas)
    # current_contour = np.zeros(img.shape[:2],np.uint8)
    current_contour = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(current_contour, contours, contour_count, (255, 255, 255), -1)

    height, width = current_contour.shape[:2]

    # all_white_pixels = []
    current_white_pixels = []

    for i in range(0, height):
      for j in range(0, width):
        if (current_contour.item(i, j) == 255):
          current_white_pixels.append([i, j])

    matrix = np.array(current_white_pixels)

    """Finding covariance matrix"""
    C = np.cov(matrix.T)

    eigenvalues, eigenvectors = np.linalg.eig(C)

    """Finding max eigenvalue"""
    # max_ev = max(eigenvalues)
    """Finding index of max eigenvalue"""
    max_index = eigenvalues.argmax(axis=0)

    """The largest eigen value gives the approximate length of the bounding
        ellipse around the largest word. If we follow the index of the largest 
        eigen value and find the eigen vectors in the column of that index,
        we'll get the x and y coordinates of it's centre."""
    y = eigenvectors[1, max_index]
    x = eigenvectors[0, max_index]

    angle = (np.arctan2(y, x)) * (180 / np.pi)
    all_angles.append(angle)
    contour_count += 1
    logging.debug(contour_count)

    logging.debug(all_angles)
    angle = np.mean(all_angles)
    logging.debug(angle)

  k = 0
  non_zero_angles = []

  for i in all_angles:
    if ((i != 0) and (i != 90.0)):
      non_zero_angles.append(i)

  logging.debug(non_zero_angles)

  rounded_angles = []
  for i in non_zero_angles:
    rounded_angles.append(np.round(i, 0))

  logging.debug(rounded_angles)
  logging.debug("mode is")
  # logging.debug(np.mode(rounded_angles))
  # angle = np.mean(non_zero_angles)
  # angle = np.mode(rounded_angles)

  mode_angle = mode(rounded_angles)[0][0]
  logging.debug(mode_angle)

  precision_angles = []
  for i in non_zero_angles:
    if (np.round(i, 0) == mode_angle):
      precision_angles.append(i)

  logging.debug('precision angles:')
  logging.debug(precision_angles)

  angle = np.mean(precision_angles)
  logging.debug('Finally, the required angle is:')
  logging.debug(angle)

  # M = cv2.getRotationMatrix2D((width/2,height/2),-(90+angle),1)
  M = cv2.getRotationMatrix2D((width / 2, height / 2), -(90 + angle), 1)
  dst = cv2.warpAffine(img, M, (width_orig, height_orig))

  # cv2.imshow('final',dst)
  cv2.imwrite('images/skewcorrected2.jpg', dst)

  return dst


def preprocess(img):
  return skew_correction(img)

# Does not work with linux:
# cv2.destroyAllWindows()


#detecting characters on image creating key points on characters.

#Detecting characters on image using keypoints



#detecting keypoints caharacter characters on the image

#this process draws keypoints on all characters available on the image

#the image to be processed is passsed in here, such that cv2.imread = 'image.png'
img = cv2.imread('image_resize.jpg') #pass the image
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(edgeThreshold=5,nfeatures=10000, scoreType=cv2.ORB_HARRIS_SCORE,scaleFactor=1.2) 
kp ,des= orb.detectAndCompute(gray,None)

img=cv2.drawKeypoints(gray,kp,None)
cv2.imwrite('processed/images/sift_keypoints.jpg',img)
    
    
# import libraries
import csv
import cv2
import pytesseract


def pre_processing(image):
    """
    This function take one argument as
    input. this function will convert
    input image to binary image
    :param image: image
    :return: thresholded image
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # converting it to binary image
    threshold_img = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # saving image to view threshold image
    cv2.imwrite('processed/images/thresholded.png', threshold_img)

    cv2.imshow('threshold image', threshold_img)
    # Maintain output window until
    # user presses a key
    cv2.waitKey(0)
    # Destroying present windows on screen
    cv2.destroyAllWindows()

    return threshold_img


def parse_text(threshold_img):
    """
    This function take one argument as
    input. this function will feed input
    image to tesseract to predict text.
    :param threshold_img: image
    return: meta-data dictionary
    """
    # configuring parameters for tesseract
    tesseract_config = r'--oem 3 --psm 6'
    # now feeding image to tesseract
    details = pytesseract.image_to_data(threshold_img, output_type=pytesseract.Output.DICT,
                                        config=tesseract_config, lang='eng')
    return details


def draw_boxes(image, details, threshold_point):
    """
    This function takes three argument as
    input. it draw boxes on text area detected
    by Tesseract. it also writes resulted image to
    your local disk so that you can view it.
    :param image: image
    :param details: dictionary
    :param threshold_point: integer
    :return: None
    """
    total_boxes = len(details['text'])
    for sequence_number in range(total_boxes):
        #if int(details['conf'][sequence_number]) > threshold_point:
            (x, y, w, h) = (details['left'][sequence_number], details['top'][sequence_number],
                            details['width'][sequence_number], details['height'][sequence_number])
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # saving image to local
    cv2.imwrite('processed/images/captured_text_area.png', image)
    # display image
    cv2.imshow('captured text', image)
    # Maintain output window until user presses a key
    cv2.waitKey(0)
    # Destroying present windows on screen
    cv2.destroyAllWindows()


def format_text(details):
    """
    This function take one argument as
    input.This function will arrange
    resulted text into proper format.
    :param details: dictionary
    :return: list
    """
    parse_text = []
    word_list = []
    last_word = ''
    for word in details['text']:
        if word != '':
            word_list.append(word)
            last_word = word
        if (last_word != '' and word == '') or (word == details['text'][-1]):
            parse_text.append(word_list)
            word_list = []

    return parse_text


def write_text(formatted_text):
    """
    This function take one argument.
    it will write arranged text into
    a file.
    :param formatted_text: list
    :return: None
    """
    with open('processed/text_detected/text_detected.txt', 'w', newline="") as file:
        csv.writer(file, delimiter=" ").writerows(formatted_text)


if __name__ == "__main__":
    # reading image from local
    image = cv2.imread('image_resize.jpg')
    # calling pre_processing function to perform pre-processing on input image.
    thresholds_image = pre_processing(image)
    # calling parse_text function to get text from image by Tesseract.
    parsed_data = parse_text(thresholds_image)
    # defining threshold for draw box
    accuracy_threshold = 30
    # calling draw_boxes function which will draw dox around text area.
    draw_boxes(thresholds_image, parsed_data, accuracy_threshold)
    # calling format_text function which will format text according to input image
    arranged_text = format_text(parsed_data)
    # calling write_text function which will write arranged text into file
    write_text(arranged_text)
