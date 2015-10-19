The project is separated into two Python 3 scripts. The first is Calibrate.py which outputs camera calibration parameters to the file “camera_params.txt”. Note that for Calibrate.py, I pulled and modified the code at the http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html, so it looks quite similar. These are my results using both (0, 0) and (1.0, 1.0) as the aperture size.

Provided images:

(1.0, 1.0) aperture size
* fovx: 50.88779098043821
* fovy: 65.45637670741117
* focal length: 1.0509188918468453
* fx: 504.441068086
* fy: 497.912224221
* principal point x: 419.154103071
* principal point y: 232.983232208
* k1: 0.0405736467461
* k2: -0.0156376898039
* p1: -0.00143055555147
* p2: 0.00744932320996

(0.0, 0.0) aperture size was the same except it had a focal length of 504.44106808648576, which makes sense because the focal length is probably returned in the same units of aperture size.

My Webcam images located in the CheckerboardImages2 folder:

* (1.0, 1.0) aperture size
* fovx: 37.51186968514057
* fovy: 62.321954777339876
* focal length: 1.4724514007986627
* fx: 1060.16500858
* fy: 1058.39174481
* principal point x: 634.384485411
* principal point y: 392.141255088
* k1: 0.097549607668
* k2: -0.196198446497
* p1: 0.00302332305484
* p2: -0.00373084910959

(0.0, 0.0) aperture size was the same except it had a focal length of 1060.165008575037.

The second portion of my project, augment.py, takes those parameters as input and an optional —-video flag if there is a supplied video. If just using the webcam, the call is python3 Augment.py camera_params.txt.
