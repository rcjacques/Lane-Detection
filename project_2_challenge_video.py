'''
@Authors: Zachary Zimits, Rene Jacques
March 13, 2019
'''

import cv2
import numpy as np
import helper
from matplotlib import pyplot as plt
import scipy.signal

#colors
cyan = (255,255,0)
black = (0,0,0)
white = (255,255,255)
red = (0,0,255)
green = (0,255,0)
blue = (255,0,0)

px_to_m = (540-420)/3.7

class App:
	'''Main Class'''

	def __init__(self):
		self.K = np.array([[  1.15422732e+03,   0.00000000e+00,   6.71627794e+02],[  0.00000000e+00,   1.14818221e+03,   3.86046312e+02],[  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
		self.dist = np.array([[-2.42565104e-01,-4.77893070e-02,-1.31388084e-03,-8.79107779e-05,2.20573263e-02]])
		self.H = helper.calibrate()

		self.color_calibration = None

		self.lane = None

		self.run()

	def histogram(self,image):
		'''Find histogram and image of histogram data for input image'''

		image[np.where(image>0)] = 1
		histogram = np.sum(image,axis=0)
		
		histogram_graph = np.zeros((image.shape),np.uint8)
		for i in range(histogram.shape[0]):
			count = (960-histogram[i]).astype(int)
			histogram_graph[count:,i]=255

		return histogram, histogram_graph

	def get_colored_lines(self,image):
		'''Process the input image to find all yellow and white lines with the desired range'''
		img_hls = cv2.cvtColor(image,cv2.COLOR_BGR2HLS)
		img_hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

		#challenge

		#rgb
		y_lower_rgb = np.array([65,131,147]) # 52 149 64
		y_upper_rgb = np.array([134,171,202]) # 131 173 255

		w_lower_rgb = np.array([0,175,138]) # 172 164 210
		w_upper_rgb = np.array([255,255,255]) # 255 255 255

		y_rgb_mask = cv2.inRange(image,y_lower_rgb,y_upper_rgb)

		w_rgb_mask= cv2.inRange(image,w_lower_rgb,w_upper_rgb)

		wy_rgb_mask = cv2.bitwise_or(y_rgb_mask,w_rgb_mask)

		###########################################################################

		#challenge

		#hls
		y_lower_hls = np.array([2,126,0]) # 0 112 133
		y_upper_hls = np.array([37,161,127]) # 118 194 205
		
		w_lower_hls = np.array([0,0,0]) # 172 184 184
		w_upper_hls = np.array([31,225,24]) # 255 255 255

		y_hls_mask = cv2.inRange(image,y_lower_hls,y_upper_hls)

		w_hls_mask= cv2.inRange(image,w_lower_hls,w_upper_hls)

		wy_hls_mask = cv2.bitwise_or(y_hls_mask,w_hls_mask)

		###########################################################################

		#hsv 

		#challenge
		y_lower_hsv = np.array([0,0,0]) # 0 129 146
		y_upper_hsv = np.array([255,255,255]) # 131 255 255

		w_lower_hsv = np.array([0,0,0]) # 164 176 203
		w_upper_hsv = np.array([255,255,255]) # 255 255 255

		y_hsv_mask = cv2.inRange(image,y_lower_hsv,y_upper_hsv)

		w_hsv_mask= cv2.inRange(image,w_lower_hsv,w_upper_hsv)

		wy_hsv_mask = cv2.bitwise_or(y_hsv_mask,w_hsv_mask)

		###########################################################################
		combine_white = cv2.bitwise_and(w_rgb_mask,w_hsv_mask,w_hls_mask)
		combine_yellow = cv2.bitwise_and(y_rgb_mask,y_hsv_mask,y_hls_mask)
		combine_mask = cv2.bitwise_or(combine_white,combine_yellow)

		white_yellow_rgb = cv2.bitwise_and(image,image,mask=wy_rgb_mask)
		white_yellow_hls = cv2.bitwise_and(img_hls,img_hls,mask=wy_hls_mask)
		white_yellow_hsv = cv2.bitwise_and(img_hsv,img_hsv,mask=wy_hsv_mask)
		combine = cv2.bitwise_and(image,image,mask=combine_mask)
		c_white = cv2.bitwise_and(image,image,mask=combine_white)
		c_yellow = cv2.bitwise_and(image,image,mask=combine_yellow)

		return white_yellow_rgb,white_yellow_hls,white_yellow_hsv,combine,c_white,c_yellow

	def process_histogram(self,graph,offset=50):
		'''Process the data in the input histogram image to find the left and right values on each peak in the histogram graph'''

		blips = np.where(graph[-offset,:] > 0) 
		new = []

		last = 0
		b = 0
		for b in blips[0]:
			if len(new) == 0:
				new.append(b)
				last = b

			if last == b-1:
				last = b 
			elif b not in new:
				new.append(last)
				new.append(b)
				last = b
		new.append(b)

		return new

	def filter_histogram(self,hist,graph,bounds,show=False):
		'''Filter input histogram graph data to simplify peaks for processing'''

		base,_ = scipy.signal.find_peaks(hist)

		peaks = []
		for b in base:
			peaks.append((b,hist[b]))
			cv2.circle(graph,(b,graph.shape[0]-hist[b]),5,green,-1)
		
		for i in range(len(bounds)-1):
			left = bounds[i]
			right = bounds[i+1]
			maxima = 0
			for p in peaks:
				if p[0] >= left and p[0] <= right:
					if maxima < p[1]:
						maxima = p[1]
			if show:
				cv2.rectangle(graph,(left,graph.shape[0]-maxima),(right,graph.shape[0]),white,-1)

		return peaks

	def polynomial(self,coef,x):
		'''Computes polynomial on range of input x values'''

		a,b,c = coef 
		return (a*x**2)+(b*x)+c

	def first_derivative(self,coef,x):
		if len(coef)==3:
			a,b,c = coef 
		else:
			return 0

		return (2*a*x)+b

	def second_derivative(self,coef,x):
		if len(coef)==3:
			a,b,c = coef
		else:
			return 0

		return (2*a)

	def calculate_radius(self,at_x,coef):
		one = self.first_derivative(coef,at_x) 
		two = self.second_derivative(coef,at_x)

		one = one if one != 0 else 1
		two = two if two != 0 else 1

		return ((1+(one)**2)**(3/2))/(two)

	def calculate_avg_radius(self,x_vals,coef):
		sum_radius = 0
		for x in x_vals:
			sum_radius += self.calculate_radius(x,coef)

		return sum_radius/len(x_vals)

	def calculate_offset(self,lane_bounds,image):
		center_x = image.shape[1]//2
		left = 0
		right = 0

		minima = 0
		for lane in lane_bounds:
			if minima == 0:	
				minima = lane 
			elif minima >= lane:
				minima = lane
			elif minima < lane:
				left = minima
				right = lane 
				break
		
		if center_x-left > right-center_x:
			left_color,right_color = green,red
		elif center_x-left < right-center_x:
			left_color,right_color = red,green
		else:
			left_color,right_color = blue,blue

		cv2.line(image,(left,image.shape[0]-600),(center_x,image.shape[0]-600),left_color,5)
		cv2.line(image,(center_x,image.shape[0]-600),(right,image.shape[0]-600),right_color,5)
		
		cv2.circle(image,(center_x,image.shape[0]-600),10,blue,-1)		
		cv2.circle(image,(left,image.shape[0]-600),10,blue,-1)
		cv2.circle(image,(right,image.shape[0]-600),10,blue,-1)
		
		return round((center_x-left)/px_to_m,2)

	def get_lane_bounds(self,dist,graph,offset=50,show_lines=False):
		'''Get boundary values '''

		# display lane line data with alternating colors
		switch = True
		lane_bounds = []
		cv2.line(graph,(300,500),(660,500),cyan,5)

		for i in range(len(dist)-1):
			# draw lines between lane and line markers
			if show_lines:
				if switch:
					cv2.line(graph,(dist[i],graph.shape[0]-offset),(dist[i+1],graph.shape[0]-offset),green,5)
					switch = False
				else:
					cv2.line(graph,(dist[i],graph.shape[0]-offset),(dist[i+1],graph.shape[0]-offset),red,5)
					switch = True

			# store lane boundaries
			if dist[i+1]-dist[i] >= 30 and dist[i+1]-dist[i] < 150:
				if abs(480-dist[i]) <= 180 and abs(480-dist[i+1]) <= 180:
					lane_bounds.append(dist[i])
					lane_bounds.append(dist[i+1])
					cv2.circle(graph,(dist[i],graph.shape[0]-offset),10,blue,-1)
					cv2.circle(graph,(dist[i+1],graph.shape[0]-offset),10,blue,-1)
		try:
			lane_bounds.append(dist[dist.index(lane_bounds[-1])+1])
		except:
			pass

		return lane_bounds

	def draw_ROIs(self,lane_bounds,graph):
		'''Draw regions of interest'''

		# draw lane ROIs and line ROIs
		lanes = []
		for i in range(len(lane_bounds)-1):
			if (i+1)%2 == 0:
				# lane ROI
				cv2.rectangle(graph,(lane_bounds[i],graph.shape[0]-300),(lane_bounds[i+1],graph.shape[0]),blue,5)
			else:
				# line ROI
				cv2.rectangle(graph,(lane_bounds[i],graph.shape[0]-300),(lane_bounds[i+1],graph.shape[0]),cyan,5)

	def fit_lane_lines(self,lane_bounds,unwarp,peak,show_lines=False,use_peak=False):
		'''Calculate polynomials that fit input lane lines'''

		lane_polys = []
		avg_radii = []

		for i in range(len(lane_bounds)-1):
			if i%2 == 0:
				if use_peak:
					step = 30
					pts = np.where(unwarp[:,peak[0]-step:peak[0]+step]>0)
					cv2.rectangle(unwarp,(peak[0]-step,0),(peak[0]+step,unwarp.shape[0]),cyan,1)
					offset = peak[0]-step
				else:
					pts = np.where(unwarp[:,lane_bounds[i]:lane_bounds[i+1]]>0)
					offset = lane_bounds[i]

				# pts is a 3 x n array where 0 x n is the Y dimension 1 x n is the X dimension and 2 x n is zero
				pts[1][:] = pts[1]+offset

				try:
					coef = np.polyfit(pts[0],pts[1],2) # flip x and y to rotate curve 90 degrees
				except: 
					coef = [0,0,0] 

				x_prime_vals = np.linspace(200,960,760) # x values are now y values so we give the corresponding range
				y_prime_vals = self.polynomial(coef,x_prime_vals)

				avg_radii.append(self.calculate_avg_radius(x_prime_vals,coef))
				
				pts = []
				for i in range(len(x_prime_vals)):
					pts.append([y_prime_vals[i],x_prime_vals[i]]) # flip x and y again to rotate back to where we want the curve to be
				pts = np.int32(pts)

				lane_polys.append(pts)

				if show_lines:
					cv2.polylines(unwarp,[pts],False,green,2)

		return lane_polys, avg_radii

	def color_lanes(self,lane_polys,unwarp_orig,unwarp,show_main=False,show_secondary=False):
		'''Color in different lanes to show lane detection'''

		colors = [green,blue]
		for i in range(len(lane_polys)-1):
			i = i if not show_main else 0
			contour = np.array([np.vstack((lane_polys[i],np.flip(lane_polys[i+1],0)))])
			try:
				if cv2.contourArea(contour) <= 200000:
					cv2.drawContours(unwarp_orig,contour,-1,colors[i],-1)
					cv2.drawContours(unwarp,contour,-1,colors[i],-1)
					self.lane = contour
				else:
					cv2.drawContours(unwarp_orig,self.lane,-1,colors[i],-1)
					cv2.drawContours(unwarp,self.lane,-1,colors[i],-1)
			except:
				pass

	def filter_peaks(self,peaks,graph):
		maximum = [0,0]
		p_sum = 0
		for p in peaks:
			if maximum[1] < p[1]:
				if abs(p[0]-480) < 100:
					maximum = p 
			p_sum += p[1] 

		cv2.line(graph,(maximum[0],graph.shape[0]-maximum[1]),(maximum[0],graph.shape[0]),red,5)

		return maximum

	def process_video_from_file(self,video_file):
		'''Main function for processing data input from a video file'''

		cap = cv2.VideoCapture(video_file)

		while cap.isOpened():
			# read video frame
			ret,frame = cap.read()
			frame_copy = frame

			self.color_calibration = int(np.mean(frame))

			# image processing
			frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
			blur = cv2.GaussianBlur(frame_gray,(5,5),0)

			dst = cv2.undistort(blur, self.K, self.dist)

			# canny = cv2.Canny(frame,50,100)
			sobel_x = cv2.Sobel(dst,cv2.CV_64F,1,0,ksize=3)
			sobel_y = cv2.Sobel(dst,cv2.CV_64F,0,1,ksize=3)

			abs_sobel_x = np.absolute(sobel_x)
			sobel_x = np.uint8(abs_sobel_x)

			abs_sobel_y = np.absolute(sobel_y)
			sobel_y = np.uint8(abs_sobel_y)

			grad = cv2.addWeighted(sobel_x,0.75,sobel_y,0.25,0)

			# crop out sky
			sky_box = np.array([[[0,0],[1280,0],[1280,470],[0,470]]],'int32')
			left_box = np.array([[[0,0],[350,0],[350,960],[0,960]]],'int32')
			right_box = np.array([[[1000,0],[1280,0],[1280,960],[1000,960]]],'int32')

			cv2.fillPoly(dst,sky_box,black)
			cv2.fillPoly(dst,left_box,black)
			cv2.fillPoly(dst,right_box,black)

			# find lane line colors
			wy_rgb,wy_hls,wy_hsv,combine,c_white,c_yellow = self.get_colored_lines(frame)

			cv2.fillPoly(wy_rgb,sky_box,black)
			cv2.fillPoly(wy_hls,sky_box,black)
			cv2.fillPoly(wy_hsv,sky_box,black)
			cv2.fillPoly(combine,sky_box,black)
			cv2.fillPoly(combine,left_box,black)
			cv2.fillPoly(combine,right_box,black)

			# unwarp frame
			unwarp = cv2.warpPerspective(combine,self.H,(960,1050))
			unwarp_yellow = cv2.warpPerspective(c_yellow,self.H,(960,1050))
			unwarp_white = cv2.warpPerspective(c_white,self.H,(960,1050))
			unwarp_orig = cv2.warpPerspective(frame,self.H,(960,1050))

			unwarp_yellow_gray = cv2.cvtColor(unwarp_yellow,cv2.COLOR_BGR2GRAY)
			y_hist,y_graph = self.histogram(unwarp_yellow_gray)

			unwarp_white_gray = cv2.cvtColor(unwarp_white,cv2.COLOR_BGR2GRAY)
			w_hist,w_graph = self.histogram(unwarp_white_gray)

			# remove white block at bottom of histogram graph
			y_graph = y_graph[:-90,:]
			w_graph = w_graph[:-90,:]

			# process histogram data
			y_dist = self.process_histogram(y_graph,offset=20)
			w_dist = self.process_histogram(w_graph,offset=20)

			# convert grayscale graph to rgb (to match array dimensions)
			# graph = cv2.cvtColor(graph,cv2.COLOR_GRAY2RGB)
			y_graph = cv2.cvtColor(y_graph,cv2.COLOR_GRAY2RGB)
			w_graph = cv2.cvtColor(w_graph,cv2.COLOR_GRAY2RGB)

			# lane_bounds = self.get_lane_bounds(dist,graph,offset=20,show_lines=True)
			y_lane_bounds = self.get_lane_bounds(y_dist,y_graph,offset=20,show_lines=False)
			w_lane_bounds = self.get_lane_bounds(w_dist,w_graph,offset=20,show_lines=False)
			w_peaks = self.filter_histogram(w_hist,w_graph,w_lane_bounds,show=False)
			
			w_max_peak = self.filter_peaks(w_peaks,w_graph)

			# self.draw_ROIs(lane_bounds,graph)
			# lane_polys, radii = self.fit_lane_lines(lane_bounds,unwarp,0,show_lines=)
			y_lane_polys, y_radii = self.fit_lane_lines(y_lane_bounds,unwarp,0,show_lines=True)
			w_lane_polys, w_radii = self.fit_lane_lines(w_lane_bounds,unwarp,w_max_peak,show_lines=True,use_peak=True)

			lane_polys = y_lane_polys+w_lane_polys

			self.color_lanes(lane_polys,unwarp_orig,unwarp,show_main=True,show_secondary=False)

			final = cv2.warpPerspective(unwarp_orig,np.linalg.inv(self.H),(frame.shape[1],frame.shape[0]))	

			final = cv2.addWeighted(final,0.25,frame,1.0,0)		

			# combine the unwarped colored lines frame with the histogram graph
			graph = cv2.bitwise_or(y_graph,w_graph)
			graph_display = np.vstack((unwarp,graph[400:,:]))

			# show radius and lane data
			font = cv2.FONT_HERSHEY_SIMPLEX
			# print(radii,px_to_m)
			try:
				radius = int(round(y_radii[0]/px_to_m))
			except:
				radius = 0
				# print('NOT ENOUGH LINES FOUND')
			direction = np.sign(radius)
			# print(direction,direction<0,radius)
			offset = round((3.7/2)-self.calculate_offset(y_lane_bounds,graph_display),4)
			cv2.putText(final,'Radius: '+(str(abs(radius))  +'m' if abs(radius)<=20000 else 'Straight'),(25,50),font,1,(0,0,255),2,cv2.LINE_AA)
			cv2.putText(final,'Vehicle is '+str(offset)+'m left of center',(25,90),font,1,(0,0,255),2,cv2.LINE_AA)

			cv2.putText(final,'Vehicle is '+('moving straight' if abs(radius)>20000 else ('turning left' if direction<0 else 'turning right')),(25,130),font,1,(0,0,255),2,cv2.LINE_AA)

			# display figures
			# cv2.imshow('orig',cv2.resize(frame,(0,0),fx=0.5,fy=0.5))
			cv2.imshow('final',cv2.resize(final,(0,0),fx=0.5,fy=0.5))
			# cv2.imshow('lane hist',cv2.resize(graph_display,(0,0),fx=0.5,fy=0.5))

			cv2.imwrite('frame_grab.png',frame)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				self.running = False
				break
 
		cap.release()
		cv2.destroyAllWindows()

	def run(self):
		'''Run main function'''
		self.running = True

		while(self.running):
			self.process_video_from_file('challenge_video.mp4')

if __name__ == '__main__':
	a = App()