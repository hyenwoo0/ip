import cv2
import numpy as np
import sys

img_file = '/Users/johyeon-u/source/ip/W3/DrawImage/wonder woman bmp.bmp'
img_color = cv2.imread(img_file, cv2.IMREAD_COLOR)
print(f" W = {img_color.shape[1]}, H = {img_color.shape[0]}")

if img_color is None:
    print('이미지를 불러오는데 실패했습니다!')
    sys.exit()


# cv2.circle(img, center, radius, color, thickness=None, lineType=None, shift=None) -> img
cv2.circle(img_color, (140,300), 100, (255, 0, 0), thickness = 2, lineType=cv2.LINE_4)
cv2.circle(img_color, (140,130), 100, (255, 0, 0), thickness = 2, lineType=cv2.LINE_AA)

cv2.imshow("Wonder Woman", img_color)

cv2.waitKey()
cv2.destroyAllWindows()
