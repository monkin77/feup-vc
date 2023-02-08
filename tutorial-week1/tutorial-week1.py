#!/usr/bin/env python
# coding: utf-8

# # Week 1: Introduction to OpenCV

# In[20]:


# Requirements for this tutorial
get_ipython().system(' pip install opencv-python')
get_ipython().system(' pip install numpy')


# In[21]:


# If you prefer, you can convert this notebook to a Python script by uncommenting the following command
# ! pip install nbconvert
# ! jupyter nbconvert --to script tutorial-week1.ipynb


# In[ ]:


import cv2
import numpy as np
import os

dataDir = './data'


# 1. Images – read, write and display; ROIs

# In[ ]:


# Opening an image
img = cv2.imread(os.path.join(dataDir, 'ml.jpg'))

# Showing the image
cv2.imshow("ml.jpg", img)

# Waiting for user to press a key to close the image
cv2.waitKey(0)

# Close the window after user pressed a key
cv2.destroyWindow("ml.jpg")


# In[ ]:


# Check image size
h, w, c = img.shape
print(f'height: {h}')
print(f'width: {w}')
print(f'channels: {c}')


# In[ ]:


# Saving image in bmp format
cv2.imwrite(os.path.join(dataDir, 'ml_new.bmp'), img)


# In[19]:


# Continue exercises 1 c) and d)

# Mouse Callback on click
def mouseCallback(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        # In OpenCV, the color channels are in Blue Green Red order
        colorsB = img[y, x, 0]
        colorsG = img[y, x, 1]
        colorsR = img[y, x, 2]

        print(f'x: {x}, y: {y}')
        print(f'BGR: ({colorsB}, {colorsG}, {colorsR})')
        # cv2.putText(img=img, text=f'BGR: ({colorsB}, {colorsG}, {colorsR})', org=(300, 300), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)

# Create named window and set mouse callback
cv2.namedWindow("ex1c")
cv2.setMouseCallback("ex1c", mouseCallback)

cv2.imshow("ex1c", img)
cv2.waitKey(0)
cv2.destroyWindow("ex1c")


# In[ ]:


# Exercise 1d)



# 2. Images – representation, grayscale and color, color spaces

# In[ ]:


# Create a white image
m = np.ones((100,200,1), np.uint8)

# Change the intensity to 100
m = m * 100

# Display the image
cv2.imshow('Grayscale image', m)
cv2.waitKey(0)
cv2.destroyWindow('Grayscale image')


# In[ ]:


# Draw a line with thickness of 5 px
cv2.line(m, (0,0), (100,200), 255, 5)
cv2.line(m, (200, 0), (0, 100), 255, 5)
cv2.imshow('Grayscale image with diagonals', m)
cv2.waitKey(0)
cv2.destroyWindow('Grayscale image with diagonals')


# In[ ]:


# Continue exercises 2 b), c), d), e) and f)


# 3. Video – acquisition and simple processing

# In[ ]:


# Define a VideoCapture Object
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

frame_nr = 0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # If frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Display the resulting frame
    cv2.imshow('webcam', frame)

    # Wait for user to press s to save frame
    if cv2.waitKey(1) == ord('s'):
        frame_name = 'frame' + str(frame_nr) + '.png'
        cv2.imwrite(frame_name, frame)
        cv2.imshow("Saved frame: " + frame_name, frame)
        cv2.waitKey(0)
        cv2.destroyWindow("Saved frame: " + frame_name)

    # Wait for user to press q to quit
    if cv2.waitKey(1) == ord('q'):
        break

    frame_nr += 1

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()


# In[ ]:


# Continue exercises 3 b), c) and d)

