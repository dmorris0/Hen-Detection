# Code by DAVIDNYARKO123
# Source: https://github.com/DAVIDNYARKO123/yolov8-silva

from ultralytics import YOLO
import numpy
import cv2
import random
import os
import shutil
import time

#Original Files for Testing
orig_model = "/Users/generuan/Downloads/yolov8n_orig.pt"
orig_path = "/Users/generuan/Downloads/video (720p).mp4"

#All Input Files
pre_trained = "/Users/generuan/Downloads/yolov8n_test.pt"
video_path = "/Users/generuan/Downloads/ch1_0808_1931.mp4"

# Deletes old predictions to load new ones. Used for space-saving.
# Code from source: https://www.geeksforgeeks.org/delete-a-directory-or-file-using-python/
save_path = "/Users/generuan/YOLOv8/Predictions"

try:
    shutil.rmtree(save_path, ignore_errors=False)
except FileNotFoundError:
    os.mkdir(save_path)
print("Creating data...")

# Color for class list, only because it causes problems without it.
#Uses color code from 0-255, (blue, green, red)
detection_colors = [(133, 255, 199)]

# Must load official model first before running personal model. 
# DOESN'T WORK WITH PERSONAL MODELS YET!
model = YOLO(orig_model)
is_trained = False


try:
    model = YOLO(pre_trained)
    is_trained = True
except ModuleNotFoundError:
    model = YOLO(orig_model)
    is_trained = False

if(is_trained):
    cap = cv2.VideoCapture(video_path)
else:
    cap = cv2.VideoCapture(orig_path)
count = 0

#Capturing time for frame of reference.
#Code from source: https://stackoverflow.com/questions/66946176/how-to-speed-up-video-capture-from-webcam-in-opencv
start_time = time.time()
end_time = start_time
elapsed_time = 0

while True:

    start_time = time.time()

    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    count = count + 1
    #  resize the frame | small frame optimise the run 
    # frame = cv2.resize(frame, (frame_wid, frame_hyt))

    # Predict on image
    image_name = "img" + str(count) + ".jpg"
    detect_params = model.predict(source=[frame], conf=0.01, save=True, save_txt = True, project="Predictions", name=image_name)
    print(detect_params)

    # Convert tensor array to numpy
    DP = detect_params[0].numpy()

    if len(DP) != 0:
        for i in range(len(detect_params[0])):
            print(i)

            boxes = detect_params[0].boxes
            box = boxes[i]  # returns one box
            clsID = box.cls.numpy()[0]
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]

            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_colors[(0)],
                3,
            )

            # Display class name and confidence
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(
                frame,
                "person"
                + " "
                + str(round(conf, 3))
                + "%",
                (int(bb[0]), int(bb[1]) - 10),
                font,
                1,
                (255, 255, 255),
                2,
            )

    # Display the resulting frame
    cv2.imshow('ObjectDetection', frame)

    # Terminate run when "Q" pressed
    if cv2.waitKey(1) == ord('q'):
        break



# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

#Reorganize images and labels without folders.
files = os.listdir(save_path)
for index, folder in enumerate(files):
    #Path is /Users/generuan/YOLOv8/Predictions
    #Index starts at 0, Folder starts at img1.jpg
    old_labels = save_path + "/" + folder + "/labels/image0.txt"
    old_image = save_path + "/" + folder + "/image0.jpg"
    new_labels = save_path + "/labels" + str(index) + ".txt"
    new_image = save_path + "/image" + str(index) + ".jpg"
    os.rename(old_labels, new_labels)
    os.rename(old_image, new_image)
    os.rmdir(save_path + "/" + folder + "/labels")
    os.rmdir(save_path + "/" + folder)

image_folder = save_path + "/images"
label_folder = save_path + "/labels"
os.mkdir(image_folder)
os.mkdir(label_folder)

#Reset files save_path (otherwise, it counts above actual number of images.)
files = os.listdir(save_path)
for file in files:
    #File starts at image0.jpg
    file_path = save_path + "/" + file
    file_name, file_ext = os.path.splitext(file)
    if(file_ext == ".jpg"):
        shutil.move(file_path, image_folder)
    elif(file_ext == ".txt"):
        shutil.move(file_path, label_folder)

end_time = time.time()
elapsed_time = end_time - start_time

print(str(round(elapsed_time, 3)) + " seconds has passed.")
print(str(count) + " images have been created.")
print("This test used the trained model: " + str(is_trained))