# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 15:49:37 2021

@author: PhamHanh
"""

import numpy as np
import cv2
import glob

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def calibrate(dirpath, prefix, image_format, square_size, width=9, height=6):
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size

    objpoints = [] 
    imgpoints = []  

    if dirpath[-1:] == '/':
        dirpath = dirpath[:-1]

    images = glob.glob(dirpath+'/' + prefix + '*.' + image_format)

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return [ret, mtx, dist, rvecs, tvecs]

def save_coefficients(mtx, dist, path):
    """ Save the camera matrix and the distortion coefficients to given path/file. """
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("K", mtx)
    cv_file.write("D", dist)
    cv_file.release()
def load_coefficients(path):
    """ Loads camera matrix and distortion coefficients. """
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    camera_matrix = cv_file.getNode("K").mat()
    dist_matrix = cv_file.getNode("D").mat()

    cv_file.release()
    return [camera_matrix, dist_matrix]


if __name__ == '__main__':
    ret, mtx, dist, rvecs, tvecs = calibrate(r'C:\Users\Hacha\.spyder-py3\ChessBoardImage', 'Image', 'jpg', 0.027, 9, 6)

    save_coefficients(mtx, dist, 'cameraCalibration.yml')
    print("Calibration is finished. RMS: ", ret)
    print("Camera matrix: ")
    print(mtx)
    print("\nDistortion coeficient:")
    print(dist)
    print("\nRoation Vector")
    print(rvecs)
    print("\nTranslation vector")
    print(tvecs)