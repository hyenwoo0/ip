import cv2
import numpy as np
import sys

img_file = '/Users/johyeon-u/source/ip/W3/DrawImage/wonder woman bmp.bmp'
img_color = cv2.imread(img_file, cv2.IMREAD_COLOR)
img_gray = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
print(f" W = {img_color.shape[1]}, H = {img_color.shape[0]}")

if img_color is None or img_gray is None:
    print('이미지를 불러오는데 실패했습니다!')
    sys.exit()

cv2.imshow("Wonder Woman(color)", img_color)
cv2.imshow("Wonder Woman(gray)", img_gray)

cv2.waitKey()
cv2.destroyAllWindows()
