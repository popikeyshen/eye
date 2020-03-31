

import cv2
import torch
from facerec import FaceRecognizer, image_to_tensor, PersonType, PersonInfo
from utils.images import hconcat_resize_min, get_int_rect

import imutils
import matplotlib.pyplot as plt







### init torch device
device = torch.device('cuda:0')

### init face detector
distance_threshold = 1.0
recognizer = FaceRecognizer(device=device, distance_threshold=distance_threshold)







im = cv2.imread('/media/popikeyshen/30c5a789-895a-4cc2-910a-3c678cc563d7/eye/2_217.jpg')
im = imutils.resize(im, height=800)

### detect faces
embeddings, boxes = recognizer.get_faces_from_frame(image_to_tensor(im))

i=0
	
for box in  boxes:

		if box is not None:
			print(box)
			### box around face
			min_x, min_y, max_x, max_y = get_int_rect(box)
			

			### crop face
			cropped_face = im[ min_y:max_y, min_x:max_x]
			i+=1
			cv2.imshow(str(i),cropped_face)

			crop_im = cv2.resize(cropped_face, (48, 48))
			#crop_im_tensor = transform(crop_im)

			#feed_imgs = Variable(torch.stack([crop_im_tensor]))
			#feed_imgs = feed_imgs #.cuda()

			#a = snet(feed_imgs)
			#print(a)

			cv2.waitKey(1)

			### calc some info about out face
			#histr = cv2.calcHist([cropped_face],[0],None,[256],[0,256]) 
  
			# show the plotting graph of an image 
			#plt.plot(histr) 
			#plt.show() 
			
			cv2.rectangle(im, (min_x, min_y), (max_x, max_y), (0,255,0), 1)

#cv2.imshow("im",im)
#k = cv2.waitKey(0)


