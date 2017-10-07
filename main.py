"""
-*- coding: utf-8 -*-
@Author  : zouzhitao
@Time    : 17-10-5 下午1:38
@File    : main.py
"""

import cv2
import numpy as np

from pso import Tracker

def get_pos(event, x, y, flag, param):
    pos, window_name,img = param
    if event == cv2.EVENT_LBUTTONDOWN:
        pos.append((x, y)) # xmin, ymin
    elif event == cv2.EVENT_LBUTTONUP:
        pos.append((x,y))
        cv2.rectangle(img, pos[-2], pos[-1], (0,255,0), 1)
        cv2.imshow(window_name, img)
        cv2.waitKey(0)


def init(window_name, img):
    """
    init first frame
    :param window_name: string of img window
    :param img: GRAY img
    :return: tracker obj
    """
    pos = []
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, get_pos, (pos, window_name, img))
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # print(pos)
    return pos


if __name__ == '__main__':
    window = 'PSO tracking'
    videoName = 'Video3.mp4'
    cap = cv2.VideoCapture(videoName)
    obj_pos = None
    track = []
    while cap.isOpened():
        if obj_pos is None:
            # 初始化对象
            ret, firstFrame = cap.read()
            if not ret:
                print('can\'t open video')
                exit(0)
            gray = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)
            obj_pos = init(window+'init', firstFrame.copy())
            # edges = cv2.Canny(gray,50,150,apertureSize=3,L2gradient=True)
            for i in range(0, len(obj_pos), 2):
                left = obj_pos[i]
                right = obj_pos[i+1]
                track.append(Tracker(left, right, gray))
        else:
            # begin tracking
            ret, frame = cap.read()
            if not ret:
                print('can\'t open video')
                exit(0)
            rec = []

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray,50,150,apertureSize=3, L2gradient=True)
            for tracker in track:
                left, right = tracker.tracking_unocc(gray)
                rec.append((left, right))
            for left, right in rec:
                cv2.rectangle(frame, left, right, (0,0,255), 1)
            cv2.imshow(window, frame)
            cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()









