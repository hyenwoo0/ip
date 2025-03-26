import cv2
import numpy as np
import sys

img_file = '/Users/johyeon-u/source/ip/W3/DrawImage/wonder woman bmp.bmp'
img_color = cv2.imread(img_file, cv2.IMREAD_COLOR)
print(f" W = {img_color.shape[1]}, H = {img_color.shape[0]}")

if img_color is None:
    print('이미지를 불러오는데 실패했습니다!')
    sys.exit()


# cv2.putText(img, text, org, fontFace, fontScale, color, thickness=None, lineType=None, bottomLeftOrigin=None) -> img
cv2.putText(img_color, "Wonder Woman", (40,350), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), thickness = 1, lineType=cv2.LINE_AA)

cv2.imshow("Wonder Woman", img_color)

cv2.waitKey()
cv2.destroyAllWindows()
