import os
import sys
import cv2



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
    results = dt.extractTags(gray)
    print(results[0].p())
    # cv2.imshow("123", img)
    # cv2.waitKey(0)
    
