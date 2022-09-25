import torch
import numpy as np
import cv2


def RunVideo(model,classes,device):
        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
        output = cv2.VideoWriter('C:/Users/ramig/Desktop/deep learning with tensorflow/Jordan Road Object Detection/Data/Output Videos/ObjectDetectionOutput_2.avi',cv2.VideoWriter_fourcc(*'MJPG'),10, (416,416)) 
        cap = cv2.VideoCapture("C:/Users/ramig/Desktop/deep learning with tensorflow/Jordan Road Object Detection/Data/Input Videos/Road_2.mp4")
        while(cap.isOpened()):
                _, frame = cap.read()
                frame = cv2.resize(frame, (416,416))
                x_shape, y_shape = frame.shape[1], frame.shape[0]
                model.to(device)
                copy_frame = [frame]
                results = model(frame)
                labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
                n = len(labels)
                for i in range(n):
                        row = cord[i]
                        if row[4] >= 0.6:
                                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                                bgr = COLORS[int(labels[i])]
                                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                                cv2.putText(frame, classes[int(labels[i])], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
                output.write(frame)
                cv2.imshow('YOLOv5 Detection '+ str(device), frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        cv2.destroyAllWindows()
        output.release()
        cap.release()

def RunCamp(model,classes,device):
        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
        cap = cv2.VideoCapture(0)
        while True:
                ret, frame = cap.read()
                frame = cv2.resize(frame, (416,416))
                x_shape, y_shape = frame.shape[1], frame.shape[0]
                model.to(device)
                copy_frame = [frame]
                results = model(frame)
                labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
                n = len(labels)
                for i in range(n):
                        row = cord[i]
                        if row[4] >= 0.6:
                                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                                bgr = COLORS[int(labels[i])]
                                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                                cv2.putText(frame, classes[int(labels[i])], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
                cv2.imshow('YOLOv5 Detection '+ str(device), frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        cv2.destroyAllWindows()
        cap.release()


if __name__ == '__main__':
        # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/ramig/Desktop/deep learning with tensorflow/Jordan Road Object Detection/Data/Trained Models/best.pt',force_reload=True)

        classes = model.names

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #device = 'cpu'
        #RunCamp(model,classes,device)
        RunVideo(model,classes,device)







