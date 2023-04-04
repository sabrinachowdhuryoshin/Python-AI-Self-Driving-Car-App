'''' standard python library '''

import os, cv2
from pathlib import Path
from random import randrange

# %%

'''' define directory paths '''

tool_path = os.path.dirname(os.path.abspath(__file__))
# print(tool_path) # debug

media_path = Path(tool_path + "\\media")
# print(media_path ) # debug

algorithm_path = Path(tool_path + "\\algorithm")
# print(algorithm_path ) # debug

''' car detection from an image '''

# %%
# choose an image to detect faces in
car_img = cv2.imread(str(media_path) + "\\car_image.jpg")
# print(car_img) # debug

# show the image 
cv2.imshow("Sabrina Chowdhury's Car Detector App", car_img)
cv2.waitKey(1000) # debug

# must convert to grayscale
grayscaled_car_img = cv2.cvtColor(car_img, cv2.COLOR_BGR2GRAY)
# print(grayscaled_car_img) # debug

# show the grayscaled image
cv2.imshow("Sabrina Chowdhury's Car Detector App", grayscaled_car_img)
cv2.waitKey(1000) # debug

# load the pre-trained car classifier
car_tracker = cv2.CascadeClassifier(str(algorithm_path) + "\\car_detection.xml")
# print(car_tracker) # debug

# detect cars
car_coordinates  = car_tracker.detectMultiScale(grayscaled_car_img)
# print(car_coordinates) # debug

# draw rectangles around the cars
for (x,y,w,h) in car_coordinates:
    cv2.rectangle(car_img, (x,y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)

# display the image with the cars
cv2.imshow("Sabrina Chowdhury's Car Detector App", car_img)
cv2.waitKey(1000)

''' car and pedestrian detection from a video '''
# %%
# load the cascade algorithm
# haarcascade algorithm only takes the gray scale images
# trained_face_data = cv2.CascadeClassifier(str(algorithm_path) + "\\haarcascade_frontalface_default.xml")
# # print(trained_face_data) # debug

# %%
sample_video = cv2.VideoCapture(str(media_path) + "\\cars_and_pedestrians.mp4")


# iterate forever over frames
while True:

    # read the current frame
    successful_frame_read, frame = sample_video.read()

    # must convert to grayscale
    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    # detect faces
    car_coordinates  = car_tracker.detectMultiScale(grayscaled_frame)
    # print(car_coordinates) # debug

    # get the face coordinates dynamically
    for (x,y,w,h) in car_coordinates:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)

    # show the video
    cv2.imshow("Sabrina Chowdhury's Car Detector App", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

sample_video.release()

# %%
print("\nCode Completed!\n")