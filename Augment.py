import cv2
import numpy as np
import glob
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *
import sys
import os.path
import getopt

currFrame = None
cap = None
fovx = None
fovy = None
fy = None
fx = None
principalX = None
principalY = None
focalLength = None
dist_co = None
cameraMatrix = None
image_shape = None
out = None
spheres = False
output = False
width = None
height = None
lightZeroPosition = [10.0, 10.0, 10.0, 1.0]
lightZeroColor = [2.5, 2.5, 2.5, 1]
near = 1
far = 500


def loadParams(params):
    global fovx, fovy, aspectRatio, principalX, principalY, focalLength, dist_co, image_shape, cameraMatrix, fx, fy
    if os.path.exists(params) and os.path.isfile(params):
        f = open(params)
        fovx = float(f.readline())
        fovy = float(f.readline())
        focalLength = float(f.readline())
        fx = float(f.readline())
        fy = float(f.readline())
        cameraMatrix = np.zeros((3, 3), np.float32)
        cameraMatrix[0, 0] = fx
        cameraMatrix[1, 1] = fy
        cameraMatrix[0, 2] = float(f.readline())
        principalX = cameraMatrix[0, 2]
        cameraMatrix[1, 2] = float(f.readline())
        principalY = cameraMatrix[1, 2]
        cameraMatrix[2, 2] = 1.0
        print("Camera Matrix")
        print(cameraMatrix)

        dist_co = [float(f.readline()), float(f.readline()), float(f.readline()), float(f.readline())]
        dist_co = np.asarray(dist_co)
    else:
        print("No parameter file.")

#
# Initialize video capture
#
def initVideoCapture(fileName):
    global cap, out, image_shape
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    if output:
        out = cv2.VideoWriter('output.avi',fourcc, 30.0, (640, 480))

    if fileName is not None:
        cap = cv2.VideoCapture(fileName)
    else:
        cap = cv2.VideoCapture(0)
    if cap is None or not cap.isOpened():
        print("Error starting capture")
        return None
    else:
        image_shape = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    return cap
#
# Start our video capture
#
def startVideoCapture():

    global cap
    cap = initVideoCapture()
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def keyboard(key, x, y):
    if key.decode("utf-8") == 'q':
        global cap
        cap.release()
        cv2.destroyAllWindows()
        exit()
    if key.decode("utf-8") == ' ':
        global spheres
        spheres = not spheres

#
# function to draw Axis
#
def drawAxis(length):

    glPushAttrib(GL_POLYGON_BIT | GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT)

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    glDisable(GL_LIGHTING)

    glBegin(GL_LINES)
    glColor3f(1, 0, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(length, 0, 0)

    glColor3f(0, 1, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(0, length, 0)

    glColor3f(0, 0, 1)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, -length)
    glEnd()

    glPopAttrib()

#
# Get the object points
#
def getObjPoints():
    objp = np.zeros((6 * 8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)
    return objp

#
# Get the object points test
#
def getObjPoints2():
    four_corners_obj = np.zeros((4,3), np.float32)
    four_corners_obj[0,0] = 0.0
    four_corners_obj[0,1] = 0.0
    four_corners_obj[1,0] = 1.0
    four_corners_obj[1,1] = 0.0
    four_corners_obj[2,0] = 0.0
    four_corners_obj[2,1] = 1.0
    four_corners_obj[3,0] = 1.0
    four_corners_obj[3,1] = 1.0
    return four_corners_obj

#
# Get the image points
#
def getImagePoints():
    ret, corners = cv2.findChessboardCorners(currFrame, (8, 6), flags=cv2.CALIB_CB_FAST_CHECK)
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        gray = cv2.cvtColor(currFrame, cv2.COLOR_BGR2GRAY)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return corners2
    return None
#
# image points test
#
def getImagePoints2():
    four_corners_img = ((600.0, 400.0), (620.0, 400.0), (600.0, 420.0), (620.0, 420.0))
    four_corners_img = np.reshape(np.asarray(four_corners_img), (4,1,2))
    return four_corners_img


#
# OpenGL display loop
#
def display():
    global fovy, aspectRatio, dist_co, cameraMatrix, currFrame, output, spheres
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)


    if currFrame is not None:
        currFrame = cv2.undistort(currFrame, cameraMatrix, dist_co)
        flippedImage = cv2.flip(currFrame, 0)

        # draw the flipped image, set depth coord to 1.0 (having issue when set exactly to 1.0)
        glDisable(GL_DEPTH_TEST)
        glDrawPixels(flippedImage.shape[1], flippedImage.shape[0], GL_BGR, GL_UNSIGNED_BYTE, flippedImage.data)
        glEnable(GL_DEPTH_TEST)

        #setup our viewport
        glViewport(0, 0, width, height)

        #setup our project matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        #set our projection matrix to perspective based on camera params

        # http://kgeorge.github.io/2014/03/08/calculating-opengl-perspective-matrix-from-opencv-intrinsic-matrix/
        print("Principal x: "+ str(principalX))
        print("Principal y: "+ str(principalY))
        print("fx:"+str(fx))
        print("fy:"+str(fy))
        print("height:"+str(height))
        print("width:"+str(width))

        glFrustum(-principalX / fx, (width - principalX) / fy, (principalY - height) / fy, principalY / fy, near, far)

        #setup our model view matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()


        ret, corners = cv2.findChessboardCorners(currFrame, (8, 6), flags=cv2.CALIB_CB_FAST_CHECK)

        if ret is True:
            objp = getObjPoints()
            corners2 = getImagePoints()
            print(corners2)
            print(corners2.shape)
            print(objp.shape)
            print(objp)
            ret, rotv, tvecs = cv2.solvePnP(objp, corners2, cameraMatrix, dist_co, None, None, 0, cv2.SOLVEPNP_ITERATIVE)

            projectedImgPts, _ = cv2.projectPoints(objp, rotv, tvecs, cameraMatrix, dist_co)
            j = 0
            for i in projectedImgPts:
                cv2.circle(currFrame, (i[0][0], i[0][1]), 2, (0, 255, 0), -1)
                j = j+1
            currFrame = cv2.flip(currFrame, 0)
            glDisable(GL_DEPTH_TEST)
            glDrawPixels(currFrame.shape[1], currFrame.shape[0], GL_BGR, GL_UNSIGNED_BYTE, currFrame.data)
            glEnable(GL_DEPTH_TEST)


            if ret is True:
                rotMat, jacobian = cv2.Rodrigues(rotv)

                matrix = np.identity(4)
                matrix[0:3, 0:3] = rotMat
                matrix[0:3, 3:4] = tvecs
                newMat = np.identity(4)
                newMat[1][1] = -1
                newMat[2][2] = -1
                matrix = np.dot(newMat, matrix)
                matrix = matrix.T
                glLoadMatrixf(matrix)
                drawAxis(1.0)

                if spheres:
                    glPushMatrix()
                    drawAxis(10.0)
                    glPopMatrix()
                    for i in range(8):
                        for j in range(6):
                            glPushMatrix()
                            glTranslatef(i, j, 0)
                            glutSolidSphere(0.30, 20, 20)
                            glPopMatrix()

                else:
                    glPushMatrix()
                    glTranslatef(3.5, 2.5, -2.5)
                    glRotatef(90, -1, 0, 0)
                    color = [1.0,0.0,0.0,1.0]
                    glLightfv(GL_LIGHT0, GL_POSITION, lightZeroPosition)
                    glLightfv(GL_LIGHT0, GL_DIFFUSE, lightZeroColor)
                    glMaterialfv(GL_FRONT,GL_DIFFUSE,color)
                    #glutSolidCube(2.0)
                    glutSolidTeapot(5.0)
                    #drawAxis(10.0)
                    glPopMatrix()
        else:
            print("No checkerboard found.")

    if output:
        newFrame = np.zeros((480, 640, 3), np.uint8)
        glReadPixels(0, 0, 640, 480, GL_BGR, GL_UNSIGNED_BYTE, newFrame)
        newFrame = cv2.flip(newFrame, 0)
        out.write(newFrame)

    glutSwapBuffers()
    glutPostRedisplay()

#
# OpenGl reshape
#
def reshape(w, h):
    glViewport(0, 0, w, h)

#
# OpenGL Idle Loop
#
def idle():
    global currFrame
    ret, frame = cap.read()
    if ret is True:
        currFrame = frame
#
# Main Camera Calibration and OpenGL loop
#
def main():
    global width, height
    args, params = getopt.getopt(sys.argv[1:], '', ['video_name='])
    args = dict(args)
    video_name = args.get('--video_name')

    loadParams(params[0])
    initVideoCapture(video_name)
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    print(image_shape)
    width = image_shape[1]
    height = image_shape[0]
    glutInitWindowSize(image_shape[1], image_shape[0])

    glutCreateWindow("OpenGL / OpenCV Example")

    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glEnable(GL_CULL_FACE)

    #we cull the front faces because my depth values are reversed from typical 0 to 1
    glCullFace(GL_FRONT)
    glShadeModel(GL_SMOOTH)
    glEnable(GL_DEPTH_TEST)

    glutDisplayFunc(display)
    glutKeyboardFunc(keyboard)
    glutReshapeFunc(reshape)
    glutIdleFunc(idle)
    glutMainLoop()


if __name__ == '__main__':
    main()