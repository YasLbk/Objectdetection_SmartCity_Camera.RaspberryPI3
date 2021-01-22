# import classes
import cv2
import numpy as np
import sys


inputfile = str(sys.argv[1])
outputfile = str(sys.argv[2])


print("Input file is ", inputfile)
print("Output file is ", outputfile)


# Load Yolo
# Weight file: it’s the trained model
# Cfg file: it’s the configuration file,settings of the algory=ithm.
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
# Extract name of all objects possible to be detected
classes = []
with open("object.fr", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))


# Loading image
img = cv2.imread(inputfile)
# img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape

# Blob it’s used to extract feature from the image and to resize them. YOLO accepts three sizes:
# 320 - 609 - 416
# The outs on line 21 it’s the result of the detection. Outs is an array that conains
# all the informations about objects detected, their position and the
# confidence about the detection.
# Detecting objects
blob = cv2.dnn.blobFromImage(
    img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Showing informations on the screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        # threshold score = 0.5
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

#  Non maximum suppresion to remove the noise in case of having more boxes for the
#  same object
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)


# extract all the informations and show them on the screen.

# Box: contain the coordinates of the rectangle sorrounding the object detected.
# Label: it’s the name of the object detected
# Confidence: the confidence about the detection from 0 to 1.
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 30), font, 1.3, color, 2)

cv2.imwrite(outputfile, img)


