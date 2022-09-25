import cv2
import os
directory = 'C:/Users/ramig/Desktop/Final Project/videos/'

for filename in os.listdir(directory):
    video = os.path.join(directory, filename)
    f = open('C:/Users/ramig/Desktop/Final Project/NameCont.txt', 'r')
    NameCont = int(f.read()) + 1
    f.close()
    vidcap = cv2.VideoCapture(video)
    currentframe = 0
    while(True):
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(currentframe*10000))
        ret,frame = vidcap.read() 
        if ret:
            name = 'C:/Users/ramig/Desktop/Final Project/images/img_' + str(NameCont) + '.jpg'
            cv2.imwrite(name, frame)
            currentframe += 1
            NameCont+=1
        else: 
            break

    vidcap.release() 
    cv2.destroyAllWindows() 

    f = open('C:/Users/ramig/Desktop/Final Project/NameCont.txt', 'w')
    f.write(str(NameCont-1))
    f.close()







