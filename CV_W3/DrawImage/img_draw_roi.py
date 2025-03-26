import cv2
import numpy as np
import sys

img_file = '/Users/johyeon-u/source/ip/W3/DrawImage/wonder woman bmp.bmp'
img_color = cv2.imread(img_file, cv2.IMREAD_COLOR)
print(f" W = {img_color.shape[1]}, H = {img_color.shape[0]}")

if img_color is None:
    print('이미지를 불러오는데 실패했습니다!')
    sys.exit()

x=105; y=100; w=65; h=75
roi = img_color[y:y+h, x:x+w]
print(roi.shape)

cv2.rectangle(img_color, (x,y), (x+w, y+h), (0, 255, 0), thickness=2)

cv2.imshow("Wonder Woman", img_color)
cv2.imshow("ROI", roi)

cv2.waitKey()
cv2.destroyAllWindows()
