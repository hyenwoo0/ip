import cv2
import numpy as np
import sys

img_file = '/Users/johyeon-u/source/ip/W3/DrawImage/wonder woman bmp.bmp'
img = cv2.imread(img_file)
print(f" W = {img.shape[1]}, H = {img.shape[0]}")

if img is None or img is None:
    print('이미지를 불러오는데 실패했습니다!')
    sys.exit()

x,y,w,h = cv2.selectROI('img', img, False)

if w>0 and h>0:
    roi = img[y:y+h, x:x+w]
    img_roi = roi.copy()
    cv2.imshow('cropped', img_roi)
    # cv2.moveWindow('cropped', 100, 200)
    cv2.imwrite('./cropped.jpg', img_roi)

cv2.imshow("img", img)
cv2.waitKey()
cv2.destroyAllWindows()
