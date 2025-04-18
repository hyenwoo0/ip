import cv2
import numpy as np
import sys

isDragging = False
x0,y0,w,h = -1, -1, -1, -1
blue, red = (255,0,0), (0,0,255)

def onMouse(event, x, y, flags, param):
    global isDragging, x0, y0, img
    if event== cv2.EVENT_LBUTTONDOWN:
        isDragging = True
        x0 = x
        y0 = y
    elif event == cv2.EVENT_MOUSEMOVE:
        if isDragging:
            img_draw = img.copy()
            cv2.rectangle(img_draw, (x0, y0), (x, y), blue, 2)
            cv2.imshow('img', img_draw)
    elif event == cv2.EVENT_LBUTTONUP:
        if isDragging:
            isDragging = False
            w = x-x0
            h = y-y0
            print(f'x:{x0}, y:{y0}, w:{w}, h:{h}')
            if w>0 and h>0:
                img_draw = img.copy()
                cv2.rectangle(img_draw, (x0, y0), (x, y), red, 2)
                cv2.imshow('img', img_draw)
                roi = img[y0:y0+h, x0:x0+w]
                cv2.imshow('cropped', roi)
                cv2.moveWindow('cropped', 100,200)
                cv2.imwrite('./cropped.jpg', roi)
                print('cropped.')
            else:
                cv2.imshow('img', img)
                print('좌측 상단에서 우측 하단으로 영역을 드래그하세요.')

img_file = '/Users/johyeon-u/source/ip/W3/DrawImage/wonder woman bmp.bmp'
img = cv2.imread(img_file)
print(f" W = {img.shape[1]}, H = {img.shape[0]}")

if img is None:
    print('이미지를 불러오는데 실패했습니다!')
    sys.exit()

cv2.imshow("img", img)
cv2.setMouseCallback('img', onMouse)

cv2.waitKey()
cv2.destroyAllWindows()
