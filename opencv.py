import cv2
import numpy as np
import imutils

cap = cv2.VideoCapture(0)  # Use 0 instead of 1 for the default camera
cap.set(3, 640)
cap.set(4, 480)

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([25, 70, 120])
    upper_yellow = np.array([30, 255, 255])

    lower_green = np.array([40, 70, 80])
    upper_green = np.array([70, 255, 255])

    lower_red = np.array([0, 50, 120])
    upper_red = np.array([10, 255, 255])

    lower_blue = np.array([90, 60, 0])
    upper_blue = np.array([121, 255, 255])

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([90, 90, 90])

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([255, 255, 255])

    mask1 = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask2 = cv2.inRange(hsv, lower_green, upper_green)
    mask3 = cv2.inRange(hsv, lower_red, upper_red)
    mask4 = cv2.inRange(hsv, lower_blue, upper_blue)
    mask5 = cv2.inRange(hsv, lower_black, upper_black)
    mask6 = cv2.inRange(hsv, lower_white, upper_white)

    cnts1 = cv2.findContours(mask1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts1 = imutils.grab_contours(cnts1)

    cnts2 = cv2.findContours(mask2.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts2 = imutils.grab_contours(cnts2)

    cnts3 = cv2.findContours(mask3.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts3 = imutils.grab_contours(cnts3)

    cnts4 = cv2.findContours(mask4.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts4 = imutils.grab_contours(cnts4)

    cnts5 = cv2.findContours(mask5.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts5 = imutils.grab_contours(cnts5)

    cnts6 = cv2.findContours(mask6.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts6 = imutils.grab_contours(cnts6)


    for c in cnts1:
        area1 = cv2.contourArea(c)
        if area1 > 5000:
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)
            M = cv2.moments(c)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
            cv2.putText(frame, "Yellow", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)

    for c in cnts2:
        area2 = cv2.contourArea(c)
        if area2 > 5000:
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)
            M = cv2.moments(c)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
            cv2.putText(frame, "Green", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)

    for c in cnts3:
        area3 = cv2.contourArea(c)
        if area3 > 5000:
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)
            M = cv2.moments(c)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
            cv2.putText(frame, "Red", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)

    for c in cnts4:
        area4 = cv2.contourArea(c)
        if area4 > 5000:
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)
            M = cv2.moments(c)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
            cv2.putText(frame, "Blue", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)

    for c in cnts5:
        area5 = cv2.contourArea(c)
        if area5 > 5000:
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)
            M = cv2.moments(c)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
            cv2.putText(frame, "black", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)
            
    for c in cnts6:
        area6 = cv2.contourArea(c)
        if area6 > 5000:
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)
            M = cv2.moments(c)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
            cv2.putText(frame, "white", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)

   

    cv2.imshow("result", frame)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
