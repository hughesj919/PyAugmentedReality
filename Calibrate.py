__author__ = 'jhughes'

import cv2
import numpy as np
import glob
import getopt
import sys

#
# Calibrate the camera given a folder name containing a set of images
# Note that a good portion of this code was pulled from the Python camera calibration tutorial:
# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
#
def calibrateCamera(folderName):

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((6 * 8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob(str(folderName+'/*'))
    print(images)

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)
        if ret is True:
            objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (8, 6), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

    cv2.destroyAllWindows()

    print("Image size:")
    print(gray.shape)
    image_shape = gray.shape

    ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None, None, None, cv2.CALIB_FIX_K3)
    print("Camera matrix...")
    print(cameraMatrix)

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    print("Total average reprojection error: ", mean_error / len(objpoints))

    fovx, fovy, focalLength, principalPoint, aspectRatio = cv2.calibrationMatrixValues(cameraMatrix, gray.shape, 0.0, 0.0)
    print("Field of view, x: "+str(fovx))
    print("Field of view, y: "+str(fovy))
    print("Focal Length: "+str(focalLength))
    print("Principal Point: "+str(principalPoint))
    print("Aspect Ratio: "+str(aspectRatio))
    print("Distortion coefficients...")
    print(dist[0][0])
    print(dist[0][1])
    print(dist[0][2])
    print(dist[0][3])

    f = open("camera_params.txt", 'w')
    f.write(str(fovx)+"\n")
    f.write(str(fovy)+"\n")
    f.write(str(focalLength)+"\n")
    f.write(str(cameraMatrix[0][0])+"\n")
    f.write(str(cameraMatrix[1][1])+"\n")
    f.write(str(cameraMatrix[0][2])+"\n")
    f.write(str(cameraMatrix[1][2])+"\n")
    f.write(str(dist[0][0])+"\n")
    f.write(str(dist[0][1])+"\n")
    f.write(str(dist[0][2])+"\n")
    f.write(str(dist[0][3]))

def main():
    args, folder_name = getopt.getopt(sys.argv[1:], '', [''])
    args = dict(args)
    print(folder_name[0])
    calibrateCamera(folder_name[0])

if __name__ == '__main__':
    main()