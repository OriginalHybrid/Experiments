# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 18:17:06 2021

@author: Himanshu
"""

# Program To Read video 
# and Extract Frames 
import cv2 
  
# Function to extract frames 
def FrameCapture(path): 
      
    # Path to video file 
    vidObj = cv2.VideoCapture(path) 
  
    # Used as counter variable 
    count = 0
  
    # checks whether frames were extracted 
    success = 1
  
    while success: 
  
        # vidObj object calls read 
        # function extract frames 
        success, image = vidObj.read() 
        #save path
        path = "E:/Data/Video/frames2/"
  
        # Saves the frames with frame-count 
        cv2.imwrite(path+"frame%d.png" % count, image) 
  
        count += 1
  
# Driver Code 
if __name__ == '__main__': 
  
    # Calling the function 
    FrameCapture("E:/Downloads/Video/THE LOVE MEDLEY - Bhage Re Man - Tu Aashiqui Hai - Mileya Mileya.mkv") 