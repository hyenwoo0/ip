import numpy as np
import cv2

src = cv2.imread("./ch4/road.jpg")
dst = src.copy()
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray, 5000, 1500, apertureSize = 5, L2gradient = True)
lines = cv2.HoughLines(canny, 0.8, np.pi / 180, 150, srn = 100, stn = 200, min_theta = 0, max_theta = np.pi)

for i in lines:
    rho, theta = i[0][0], i[0][1]
    a, b = np.cos(theta), np.sin(theta)
    x0, y0 = a*rho, b*rho

    print(f"x0:{x0}, y0:{y0}")

    scale = src.shape[0] + src.shape[1]

    x1 = int(x0 + scale * -b)
    y1 = int(y0 + scale * a)
    x2 = int(x0 - scale * -b)
    y2 = int(y0 - scale * a)

    cv2.line(dst, (int(x1), int(y1)), (x2, y2), (0, 0, 255), 2)
    # cv2.circle(dst, (int(min(max(x0,10), dst.shape[1]-10)), int(min(max(y0,10), dst.shape[0]-10))), 3, (255, 0, 0), 5, cv2.FILLED)
print("number of lines: ",len(lines))
cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()