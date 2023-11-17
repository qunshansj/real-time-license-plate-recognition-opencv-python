
import HyperLPRLite as pr
import cv2
import numpy as np
grr = cv2.imread("images_rec/2_.jpg")
model = pr.LPR("model/cascade.xml","model/model12.h5","model/ocr_plate_all_gru.h5")
for pstr,confidence,rect in model.SimpleRecognizePlateByE2E(grr):
        if confidence>0.7:
            image = drawRectBox(grr, rect, pstr+" "+str(round(confidence,3)))
            print "plate_str:"
            print pstr
            print "plate_confidence"
            print confidence

cv2.imshow("image",image)
cv2.waitKey(0)
