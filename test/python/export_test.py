import os
import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("input img")
        quit()    

    build_dir = os.path.normpath(os.path.dirname(os.path.realpath(__file__)) + os.sep + ".." + os.sep + ".." + os.sep + "build")
    print(build_dir)
    sys.path.append(build_dir)

    import py_ethztag as tag

    dt = tag.TagDetector()
    # print(d.good)
    # rt = d.getRelativeTransform(0.1, 500, 500, 320, 240)
    # print(rt)

    image_filename = sys.argv[1]
    img = cv2.imread(image_filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # print(img.shape)
    detections = dt.extractTags(gray)
    det_dict = {det.id: det for det in detections}
    
    # quit()
    plt.figure()
    plt.imshow(gray, 'gray')
    plt.plot(det_dict[0].p()[0, 0], det_dict[0].p()[0, 1], 'o')
    plt.plot(det_dict[5].p()[1, 0], det_dict[5].p()[1, 1], 'o')
    plt.plot(det_dict[30].p()[3, 0], det_dict[30].p()[3, 1], 'o')
    plt.plot(det_dict[35].p()[2, 0], det_dict[35].p()[2, 1], 'o')
    plt.show()
    # print(results[0].p())
    # cv2.imshow("123", img)
    # cv2.waitKey(0)
    
