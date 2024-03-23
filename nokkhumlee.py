#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

# โหลด pre-trained Haar cascade สำหรับ face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# โหลด pre-trained Haar cascade สำหรับการตรวจจับแมส
mask_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mask.xml')

# โหลด pre-trained Haar cascade สำหรับการตรวจจับแว่น
glasses_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

# โมเดล pre-trained สำหรับการตรวจจับเพศ
gender_net = cv2.dnn.readNetFromCaffe('models/deploy_gender.prototxt', 'models/gender_net.caffemodel')

def image_callback(msg):
    bridge = CvBridge()
    # แปลงข้อมูลภาพจาก ROS Image message เป็น OpenCV image
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    # แปลงภาพเป็น grayscale เนื่องจาก Haar cascades ทำงานกับภาพแบบ grayscale
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    
    # ตรวจจับใบหน้าในภาพ
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # วาดสี่เหลี่ยมรอบใบหน้า
        cv2.rectangle(cv_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # ในส่วนของใบหน้าที่ตรวจพบ จะตรวจจับแมส
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = cv_image[y:y+h, x:x+w]
        
        masks = mask_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
        
        for (mx, my, mw, mh) in masks:
            # วาดสี่เหลี่ยมรอบแมส
            cv2.rectangle(roi_color, (mx, my), (mx+mw, my+mh), (0, 255, 0), 2)
            
        glasses = glasses_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
        
        for (gx, gy, gw, gh) in glasses:
            # วาดสี่เหลี่ยมรอบแว่น
            cv2.rectangle(roi_color, (gx, gy), (gx+gw, gy+gh), (0, 0, 255), 2)
            
        # สำหรับการตรวจจับเพศ
        blob = cv2.dnn.blobFromImage(roi_color, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = "Male" if gender_preds[0][0] > gender_preds[0][1] else "Female"
        
        # แสดงเพศบนภาพ
        label = "{}".format(gender)
        cv2.putText(cv_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # แสดงภาพผลลัพธ์
    cv2.imshow('Detection', cv_image)
    cv2.waitKey(3)

def camera_detection():
    rospy.init_node('camera_detection_node', anonymous=True)
    rospy.Subscriber('/camera/image_raw', Image, image_callback)
    rospy.spin()

if __name__ == '__main__':
    camera_detection()
