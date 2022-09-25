# rm *.png

from glob import glob                                                           
import cv2 
pngs = glob('C:/Users/ramig/Desktop/Final Project/images/*.png')


for j in pngs:
    img = cv2.imread(j)
    cv2.imwrite(j[:-3] + 'jpg', img)