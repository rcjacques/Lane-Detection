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

px_to_m = (902-765)/3.7

class App:
	'''Main Class'''

	def __init__(self):
		self.K = np.array([[  1.15422732e+03,   0.00000000e+00,   6.71627794e+02],[  0.00000000e+00,   1.14818221e+03,   3.86046312e+02],[  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
		self.dist = np.array([[-2.42565104e-01,-4.77893070e-02,-1.31388084e-03,-8.79107779e-05,2.20573263e-02]])
		self.H = helper.calibrate()

		self.color_calibration = None

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
		'''Process the input image to find all yellow_upper_bound llow and white lines with the desired range'''

		#project video
		yellow_lower_bound = np.array([0,0,200])
		yellow_upper_bound = np.array([150,255,255])
		yellow_mask = cv2.inRange(image,yellow_lower_bound,yellow_upper_bound)

		white_lower_bound = np.array([200,200,200])
		white_upper_bound = np.array([255,255,255])
		white_mask= cv2.inRange(image,white_lower_bound,white_upper_bound)

		white_yellow_mask = cv2.bitwise_or(yellow_mask,white_mask)

		white_yellow_frame = cv2.bitwise_and(image,image,mask=white_yellow_mask)

		return white_yellow_frame

	def process_histogram(self,graph,offset=50):
		'''Process the data in the input histogram image to find the left and right values on each peak in the histogram graph'''

		blips = np.where(graph[-offset,:] > 0) #changed from -50 to -1 to try filtering histogram
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

		for lane in lane_bounds:
			if center_x-lane <= 0:
				left = lane_bounds[lane_bounds.index(lane)-1]
				right = lane
				break
		
		if center_x-left > right-center_x:
			left_color,right_color = green,red
		elif center_x-left < right-center_x:
			left_color,right_color = red,green
		else:
			left_color,right_color = blue,blue
		
		return round((center_x-left)/px_to_m,2)

	def get_lane_bounds(self,dist,graph,offset=50,show_lines=False):
		'''Get boundary values '''

		# display lane line data with alternating colors
		switch = True
		lane_bounds = []
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
			if dist[i+1]-dist[i] >= 80 and dist[i+1]-dist[i] < 150:
				if len(lane_bounds) == 0:
					lane_bounds.append(dist[i-1])
				lane_bounds.append(dist[i])
				lane_bounds.append(dist[i+1])
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

	def fit_lane_lines(self,lane_bounds,unwarp,show_lines=False):
		'''Calculate polynomials that fit input lane lines'''

		lane_polys = []
		avg_radii = []
		for i in range(len(lane_bounds)-1):
			if i%2 == 0:
				pts = np.where(unwarp[:,lane_bounds[i]:lane_bounds[i+1]]>0)

				# pts is a 3 x n array where 0 x n is the Y dimension 1 x n is the X dimension and 2 x n is zero
				pts[1][:] = pts[1]+lane_bounds[i]

				try:
					coef = np.polyfit(pts[0],pts[1],2) # flip x and y to rotate curve 90 degrees
				except:
					coef = [0,0,0]

				x_prime_vals = np.linspace(0,960,960) # x values are now y values so we give the corresponding range
				y_prime_vals = self.polynomial(coef,x_prime_vals)

				avg_radii.append(self.calculate_avg_radius(x_prime_vals,coef))
				
				pts = []
				for i in range(len(x_prime_vals)):
					pts.append([y_prime_vals[i],x_prime_vals[i]]) # flip x and y again to rotate back to where we want the curve to be
				pts = np.int32(pts)

				lane_polys.append(pts)

				if show_lines:
					cv2.polylines(unwarp,[pts],False,green,5)

		return lane_polys, avg_radii

	def color_lanes(self,lane_polys,unwarp_orig,unwarp,show_main=False,show_secondary=False):
		'''Color in different lanes to show lane detection'''

		colors = [green,blue]
		for i in range(len(lane_polys)-1):
			i = i if not show_main else 0
			contour = np.array([np.vstack((lane_polys[i],np.flip(lane_polys[i+1],0)))])
			try:
				cv2.drawContours(unwarp_orig,contour,-1,colors[i],-1)
				cv2.drawContours(unwarp,contour,-1,colors[i],-1)
			except:
				pass

	def process_video_from_file(self,video_file):
		'''Main function for processing data input from a video file'''

		cap = cv2.VideoCapture(video_file)

		while cap.isOpened():
			# read video frame
			ret,frame = cap.read()

			# image processing
			frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
			blur = cv2.GaussianBlur(frame_gray,(5,5),0)

			dst = cv2.undistort(blur, self.K, self.dist)

			# crop out sky
			sky_box = np.array([[[0,0],[1280,0],[1280,360],[0,360]]],'int32')

			cv2.fillPoly(dst,sky_box,black)

			# find lane line colors
			wy_frame = self.get_colored_lines(frame)

			cv2.fillPoly(wy_frame,sky_box,black)

			# unwarp frame
			unwarp = cv2.warpPerspective(wy_frame,self.H,(960,1050))
			unwarp_gray = cv2.cvtColor(unwarp,cv2.COLOR_BGR2GRAY)
			unwarp_orig = cv2.warpPerspective(frame,self.H,(960,1050))
			
			# find histogram
			wy_gray = cv2.cvtColor(wy_frame,cv2.COLOR_BGR2GRAY)
			hist,graph = self.histogram(unwarp_gray)

			# remove white block at bottom of histogram graph
			graph = graph[:-90,:]

			# process histogram data
			dist = self.process_histogram(graph,offset=50)
			
			# convert grayscale graph to rgb (to match array dimensions)
			graph = cv2.cvtColor(graph,cv2.COLOR_GRAY2RGB)

			lane_bounds = self.get_lane_bounds(dist,graph,offset=50,show_lines=True)
			self.filter_histogram(hist,graph,lane_bounds)

			self.draw_ROIs(lane_bounds,graph)
			
			lane_polys, radii = self.fit_lane_lines(lane_bounds,unwarp,show_lines=True)

			self.color_lanes(lane_polys,unwarp_orig,unwarp,show_main=True)

			final = cv2.warpPerspective(unwarp_orig,np.linalg.inv(self.H),(frame.shape[1],frame.shape[0]))	

			final = cv2.addWeighted(final,0.25,frame,1.0,0)		

			# combine the unwarped colored lines frame with the histogram graph
			graph_display = np.vstack((unwarp,graph[400:,:]))

			# show radius and lane data
			font = cv2.FONT_HERSHEY_SIMPLEX

			try:
				radius = int(round(radii[0]/px_to_m))
			except:
				radius = 0
				print('NOT ENOUGH LINES FOUND')
			direction = np.sign(radius)
			offset = round((3.7/2)-self.calculate_offset(lane_bounds,graph_display),2)
			cv2.putText(final,'Radius: '+(str(abs(radius))+'m' if abs(radius)<=20000 else 'Straight'),(25,50),font,1,(0,0,255),2,cv2.LINE_AA)
			cv2.putText(final,'Vehicle is '+str(offset)+'m left of center',(25,90),font,1,(0,0,255),2,cv2.LINE_AA)

			cv2.putText(final,'Vehicle is '+('moving straight' if abs(radius)>20000 else ('turning left' if direction<0 else 'turning right')),(25,130),font,1,(0,0,255),2,cv2.LINE_AA)

			# display figures
			# cv2.imshow('orig',cv2.resize(frame,(0,0),fx=0.5,fy=0.5))
			cv2.imshow('final',cv2.resize(final,(0,0),fx=0.5,fy=0.5))
			# cv2.imshow('lane hist',cv2.resize(graph_display,(0,0),fx=0.5,fy=0.5))

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		cap.release()
		cv2.destroyAllWindows()

	def run(self):
		'''Run main function'''

		self.process_video_from_file('project_video.mp4')

if __name__ == '__main__':
	a = App()