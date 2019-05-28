import numpy as np
import cv2

red = [0,0,255]
green= [0,255,0]
blue = [255,0,0]

def calibrate(image=None,show=False):
	c1 = [550,495]
	c2 = [765,495]
	c3 = [902,572]
	c4 = [458,572]

	c_points = np.array([c1,c2,c3,c4])
	
	w1 = [420,600]
	w2 = [540,600]
	w3 = [540,900]
	w4 = [420,900]

	if show:
		color = red
		cv2.circle(image,(c1[0],c1[1]),5,color,-1)
		cv2.circle(image,(c2[0],c2[1]),5,color,-1)
		cv2.circle(image,(c3[0],c3[1]),5,color,-1)
		cv2.circle(image,(c4[0],c4[1]),5,color,-1)

	w_points = np.array([w1,w2,w3,w4])

	H, status = cv2.findHomography(c_points,w_points)

	return H