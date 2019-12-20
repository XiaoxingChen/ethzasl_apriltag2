import os
import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt

def PolygonScale(points, scale=1.0):
    weight_center = points.sum(axis=0) / len(points)
    scaled_points = (points - weight_center) * scale + weight_center
    return scaled_points


# link of perspective transform: https://github.com/opencv/opencv/blob/11b020b9f9e111bddd40bffe3b1759aa02d966f0/modules/imgproc/src/imgwarp.cpp#L3001

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")

	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
 
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
 
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
 
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
 
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
	# return the warped image
	return warped

def BoardExtraction(gray_img):
    import py_ethztag as tag
    dt = tag.TagDetector()

    detections = dt.extractTags(gray_img)
    print("detect {} tags".format(len(detections)))
    det_dict = {det.id: det for det in detections}
    board_corners = np.array([
        [det_dict[0].p()[0, 0], det_dict[0].p()[0, 1]],
        [det_dict[5].p()[1, 0], det_dict[5].p()[1, 1]],
        [det_dict[35].p()[2, 0], det_dict[35].p()[2, 1]],
        [det_dict[30].p()[3, 0], det_dict[30].p()[3, 1]]])

    board_corners = PolygonScale(board_corners, 1.1)
    board = four_point_transform(gray_img, board_corners)
    return board 

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
    # detections = dt.extractTags(gray)
    # det_dict = {det.id: det for det in detections}
    # p_outs = np.array([
    #     [det_dict[0].p()[0, 0], det_dict[0].p()[0, 1]],
    #     [det_dict[5].p()[1, 0], det_dict[5].p()[1, 1]],
    #     [det_dict[35].p()[2, 0], det_dict[35].p()[2, 1]],
    #     [det_dict[30].p()[3, 0], det_dict[30].p()[3, 1]]])

    # print(p_outs)
    # # quit()

    # p_outs = PolygonScale(p_outs, 1.1)

    # p4_tf = four_point_transform(gray, p_outs)
    # p_outs.append(p_outs[0])
    # p_outs = np.vstack([p_outs, p_outs[0]])
    
    # p_outs = np.array(p_outs)
    
    p4_tf = BoardExtraction(gray)
    # quit()
    plt.figure()
    plt.imshow(gray, 'gray')
    # plt.plot(det_dict[0].p()[0, 0], det_dict[0].p()[0, 1], 'o')
    # plt.plot(det_dict[5].p()[1, 0], det_dict[5].p()[1, 1], 'o')
    # plt.plot(det_dict[30].p()[3, 0], det_dict[30].p()[3, 1], 'o')
    # plt.plot(det_dict[35].p()[2, 0], det_dict[35].p()[2, 1], 'o')
    # plt.plot(p_outs[:,0], p_outs[:,1], 'o-')
    
    # print(results[0].p())
    # cv2.imshow("123", img)
    # cv2.waitKey(0)

    plt.figure()
    plt.imshow(p4_tf, 'gray')
    plt.show()
    
