# Open CV : Object detection using Python
## Introduction:
### Object detection is an important task, yet challenging vision task. It is a critical part of many applications such as image search, image auto-annotation and scene understanding, object tracking. Moving object tracking of video image sequences was one of the most important subjects in computer vision. It had already been applied in many computer vision fields, such as smart video surveillance (Arun Hampapur 2005), artificial intelligence, military guidance, safety detection and robot navigation, medical and biological application. In recent years, a number of successful single-object tracking system appeared, but in the presence of several objects, object detection becomes difficult and when objects are fully or partially occluded, they are obtruded from the human vision which further increases the problem of detection.. The proposed MLP based object tracking system is made robust by an optimum selection of unique features and also by implementing the Adaboost strong classification method.
## Input :
### This is a sample image we feed to the algorithm and expect our algorithm to detect and identify objects in the image and label them according to the class assigned to it.
![image](https://github.com/user-attachments/assets/a6911ea7-66c7-45d3-8edd-7daffbb60b41)


### ImageAI provides many more features useful for customization and production capable deployments for object detection tasks. Some of the features supported are :

	Custom Objects Detection: Using a provided CustomObject class, you can tell the detection class to report detections on one or a few number of unique objects

	Detection Speeds: You can reduce the time it takes to detect an image by setting the speed of detection speed to “fast”, “faster” and “fastest”.

	Input Types: You can specify and parse in file path to an image, Numpy array or file stream of an image as the input image .

	Output Types: You can specify that the detectObjectsFromImage function should return the image in the form of a file or Numpy array .

## Program :
```
import cv2

import numpy as np
import time 

np.random.seed(20)
class Detector:
    def __init__(self, videoPath, configPath, modelPath, classesPath):
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath =modelPath
        self.classesPath = classesPath
    

        self.net = cv2.dnn.DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320,320)
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        self.readClasses()

    def readClasses(self):
       with open(self.classesPath, 'r') as f:
           self.classesList = f.read().splitlines()
       self.classesList.insert(0, '__Background__')

       self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))

       # print(self.classesList)   

    def onVideo(self):
        cap = cv2.VideoCapture(self.videoPath)

        if (cap.isOpened()==False):
            print("Error opening file...")
            return

        (success, image) = cap.read()

        startTime = 0

        while success:
            currentTime = time.time()
            fps = 1/(currentTime - startTime)
            startTime = currentTime

            classLabelIDs, confidences, bboxs = self.net.detect(image, confThreshold = 0.4)

            bboxs = list(bboxs)
            confidences = list(np.array(confidences).reshape(1,-1)[0])
            confidences = list(map(float, confidences))

            bboxIdx = cv2.dnn.NMSBoxes(bboxs, confidences, score_threshold = 0.5, nms_threshold = 0.2)

            if len(bboxIdx) !=0:
                for i in range(0, len(bboxIdx)):

                    bbox = bboxs[np.squeeze(bboxIdx[i])]
                    classConfidence = confidences[np.squeeze(bboxIdx[i])]
                    classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxIdx[i])])
                    classLabel = self.classesList[classLabelID]
                    classcolor = [int(c) for c in self.colorList[classLabelID]]

                    displayText = "{}: {:.2f}".format(classLabel, classConfidence)

                    x,y,w,h = bbox

                    cv2.rectangle(image, (x,y), (x+w, y+h), color=classcolor, thickness=1)
                    cv2.putText(image, displayText, (x,y-10), cv2.FONT_HERSHEY_PLAIN, 1, classcolor, 2)

                    lineWidth = min(int(w * 0.3), int(h * 0.3)) 

                    cv2.line(image, (x,y), (x +lineWidth, y), classcolor, thickness= 5)
                    cv2.line(image, (x,y), (x, y + lineWidth), classcolor, thickness= 5)

                    cv2.line(image, (x + w,y), (x + w - lineWidth, y), classcolor, thickness= 5)
                    cv2.line(image, (x + w,y), (x + w, y + lineWidth), classcolor, thickness= 5)

                    cv2.line(image, (x,y + h), (x +lineWidth, y + h), classcolor, thickness= 5)
                    cv2.line(image, (x,y + h), (x, y + h - lineWidth), classcolor, thickness= 5)

                    cv2.line(image, (x + w,y + h), (x + w - lineWidth, y + h), classcolor, thickness= 5)
                    cv2.line(image, (x + w,y + h), (x + w, y + h - lineWidth), classcolor, thickness= 5)

            cv2.putText(image, "FPS: " + str(int(fps)), (20,70), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
            cv2.imshow("Result", image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            (success,image) = cap.read()  
        cv2.destroyAllWindows()          







# MAIN PYTHON CODE:

from Detector import *
import os

def main():
    videoPath =  0

    configPath = os.path.join("model_data", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelPath = os.path.join("model_data", "frozen_inference_graph.pb")
    classesPath = os.path.join("model_data", "coco.names")

    detector = Detector(videoPath, configPath, modelPath, classesPath)
    detector.onVideo()

if __name__ == '__main__': 
    main()
```
## Output :

### As expected our algorithm identifies the objects by its classes ans assigns each object by its tag and has dimensions on detected image.
![image](https://github.com/user-attachments/assets/eee84fa2-1549-429d-bea5-eafb594e58ee)

## Summary:
### The object detection report encompasses various stages, starting with data collection and concluding with model deployment and maintenance. The process involves gathering a diverse dataset, annotating images, and preprocessing data for optimal performance. Model selection, training, validation, and testing follow, ensuring a robust and accurate detection model. Post-processing techniques refine results, and integration into the target system precedes optimization for efficiency. Deployment in the intended environment marks a crucial stage, with ongoing monitoring and maintenance to adapt to evolving data patterns. The report underscores the comprehensive methodology, emphasizing the significance of each step in achieving successful object detection.



