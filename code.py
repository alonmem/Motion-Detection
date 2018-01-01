import cv2
import numpy as np
from math import sqrt
import os
import time
import sys

def convertTo1(img):  # replace 255 with 1
  img2 = np.copy(img)
  for i in range(img2.shape[0]):  
    for j in range(img2.shape[1]):
      if img2[i,j] ==255:
        img2[i,j] = 1
  return img2


def col_sum(arr, col):  # reutrn column sum
    Sum = 0
    for row in arr:
        Sum += row[col]
    return Sum


def crop(img, Type):  # get binary image and crop to most of white pixels
  th = 0.98   # 98% of white pixels
  img1 = convertTo1(img)  # 255 -> 1
  if Type=='MHI':
    th = 0.7
    img1 = img
    
  whiteNum = np.sum(img1) # number of white pixels in the frame
  
  min_row = 0
  max_row = img1.shape[0]
  min_col = 0
  max_col = img1.shape[1]

  sums = [sum(img1[min_row]), sum(img1[max_row-1]),
          col_sum(img1, min_col), col_sum(img1, max_col-1)]
  
  while np.sum(img1[min_row:max_row, min_col:max_col]) > (whiteNum * th):   # while above 98%
    val = min(sums)
    ind = sums.index(val)
    
    if ind == 0:
      min_row += 1
      sums[ind] = sum(img1[min_row])
        
    if ind == 1:
      max_row -= 1
      sums[ind] = sum(img1[max_row])
        
    if ind == 2:
      min_col += 1
      sums[ind] = col_sum(img1, min_col)
        
    if ind == 3:
      max_col -= 1
      sums[ind] = col_sum(img1, max_col)

  return img[min_row:max_row, min_col:max_col]

def generate_MEI(path):
  vidcap = cv2.VideoCapture(path)
  (success, image) = vidcap.read()
  success = True

  MEI = np.array([])
  frames = []

  while success:
    (success, image) = vidcap.read()
    if not success:
        break
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to gray scale

    if success:
      frames.append(gray_image)
      
  if len(frames)%2 == 1:
    del frames[-1]
    
  first = True
  for img in frames:
    if first:
      MEI = np.copy(img)
      first = False
      continue

    MEI = cv2.absdiff(MEI, img)   # add new image to the MEI

  #convert to binary image (with 50 threshold)
  for i in range(MEI.shape[0]):  
    for j in range(MEI.shape[1]):  
      if MEI[i, j] < 50:
        MEI[i, j] = 0
      else:
        MEI[i, j] = 255

  #cv2.imshow('MEI', crop(MEI, 'MEI'))
  #cv2.imwrite('walking/MEI/walk3.BMP', crop(MEI, 'MEI'))            
  return crop(MEI,'MEI')

def generate_MHI(path):
  vidcap = cv2.VideoCapture(path)
  (success, prev) = vidcap.read()
  success = True

  MHI = None
  temp_MHI = np.zeros((prev.shape[:2]), np.float32)
  duration = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
  timestamp = 0
  
  while success:
    (success, curr) = vidcap.read()
    if not success:
        break
      
    
    dif = cv2.absdiff(curr, prev)
    gray = cv2.cvtColor(dif, cv2.COLOR_BGR2GRAY)  # convert to gray scale
    
    ret, mask = cv2.threshold(gray, 50, 1, cv2.THRESH_BINARY)  # mask and
    timestamp += 1

    cv2.motempl.updateMotionHistory(mask, temp_MHI, timestamp, duration)

    MHI = np.uint8(np.clip((temp_MHI-(timestamp-duration)) / duration, 0, 1)*255)

    prev = curr.copy()

  #cv2.imshow('a', crop(MHI, 'MHI'))
  #cv2.imwrite('walking/MHI/walk3.BMP', crop(MHI, 'MHI'))
  return crop(MHI,'MHI')

def euclidean_distance(img1, img2):
  distance = float(0)
  img2 = cv2.resize(img2, (img1.shape[1],img1.shape[0]))
  img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # convert to gray scale
  #img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # convert to gray scale

  for i in range(img1.shape[0]):  
    for j in range(img1.shape[1]):
      distance += (img1[i,j]/255 - img2[i,j]/255)**2
      
  return sqrt(distance)

def MEI_distance(img1,img2):
  distance = float(0)
  img2 = cv2.resize(img2, (img1.shape[1],img1.shape[0]))
  img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # convert to gray scale
  # convert images to 0's and 1's
  img1_1 = np.copy(convertTo1(img1))
  img2_1 = np.copy(convertTo1(img2))

  return np.sum(img1_1 - img2_1)


def regocnizeAction(path):
  MHI_min_distance = float("inf")
  MHI_min_path = None

  MEI_min_distance = float("inf")
  MEI_min_path = None
  
  MHI = generate_MHI(path)
  MEI = generate_MEI(path)

  paths = ["arms-wave/", "sit-down/", "walking/", "jumping/"]
  Type = ["MHI", "MEI"]

  for d in paths:
    for t in Type:
      for file in os.listdir(str(d+t)):
        if "BMP" in file:
          if t=="MHI":
            comp_MHI = cv2.imread(str(d+t)+'/'+str(file))
            dist = euclidean_distance(comp_MHI, MHI)
            if dist<MHI_min_distance:    # found current min
              MHI_min_distance = dist
              MHI_min_path = file
          if t=="MEI":
            comp_MEI = cv2.imread(str(d+t)+'/'+str(file))
            dist = MEI_distance(comp_MEI, MEI)
            if dist<MEI_min_distance:    # found current min
              MEI_min_distance = dist
              MEI_min_path = file
          
  print("Action regocnized using MHI: ")
  if "arm" in MHI_min_path:
    print("arms wave!")
  if "sit" in MHI_min_path:
    print("sitting down!")
  if "walk" in MHI_min_path:
    print("walking!")
  if "jump" in MHI_min_path:
    print("jumping!")

  print("Action regocnized using MEI: ")
  if "arm" in MEI_min_path:
    print("arms wave!")
  if "sit" in MEI_min_path:
    print("sitting down!")
  if "walk" in MEI_min_path:
    print("walking!")
  if "jump" in MEI_min_path:
    print("jumping!")


if __name__ == "__main__":
  #generate_MEI('walking/walk3.MOV')
  #generate_MHI('walking/walk3.MOV')
  
  start = time.time()
  regocnizeAction('test/test1.MOV')
  print("\nrecognition took " , time.time()-start, " seconds.")
