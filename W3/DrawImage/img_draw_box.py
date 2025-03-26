import cv2
import numpy as np
import sys

img_file = '/Users/johyeon-u/source/ip/W3/DrawImage/wonder woman bmp.bmp'
img_color = cv2.imread(img_file, cv2.IMREAD_COLOR)
print(f" W = {img_color.shape[1]}, H = {img_color.shape[0]}")

if img_color is None:
    print('이미지를 불러오는데 실패했습니다!')
    sys.exit()

# cv2.rectangle(img, pt1(x1, y1), pt2(x2, y2), color(B,G,R), thickness=None, lineType=None, shift=None) -> img
cv2.rectangle(img_color, (40,30), (100, 200), (255, 0, 0), thickness = 2, lineType=cv2.LINE_4)
cv2.rectangle(img_color, (170, 20), (260, 120), (255, 0, 0), thickness = 2, lineType=cv2.LINE_AA)

# cv2.rectangle(img, rec(x,y,w,h), color(B,G,R), thickness=None, lineType=None, shift=None) -> img
cv2.rectangle(img_color, (20, 10, 100, 200), (0, 255, 0), thickness = 2, lineType=cv2.LINE_4)
cv2.rectangle(img_color, (170, 150, 100, 150), (0, 255, 0), thickness = 2, lineType=cv2.LINE_AA)

cv2.imshow("Wonder Woman", img_color)

cv2.waitKey()
cv2.destroyAllWindows()
