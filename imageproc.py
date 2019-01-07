import cv2, os
from numpy import *
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA, RandomizedPCA
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

cascadeLocation = "haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascadeLocation)
rows = 0;
column = 0;
def prepare_dataset(directory):
	paths = [os.path.join(directory, filename) for filename in os.listdir(directory)]
	images = []
	labels = []
	row = 45		#140
	col = 45
	rows = 1000
	column = 1000
	for image_path in paths:
		image_pil = Image.open(image_path).convert('L')
		image = np.array(image_pil, 'uint8')
		if os.path.split(image_path)[1][0:4] == 'male':
			nbr = 0;
		else:
			nbr = 1;
		faces = faceCascade.detectMultiScale(image)
		#discard images with more than one face or no face detected
		if(len(faces) ==0):
			continue
		if(faces.size != 4):
			continue;
		#print(faces.size)
		#face detect and crop and resize 45x45
		for (x,y,w,h) in faces:
			images.append(cv2.resize(image[y:y+h,x:x+w] , dsize =(45,45)))
			labels.append(nbr)
			#column = min(col,column);
			#rows = min(rows,row);
			#print(rows,column)
			cv2.imshow("Reading Faces ",cv2.resize(image[y:y+h,x:x+w] , dsize =(45,45)))
			cv2.waitKey(5)
	return images,labels, row, col

directory = 'yalefaces_10' # directory name containing training images
images, labels, row, col = prepare_dataset(directory)
n_components = 150 # #component in PCA
cv2.destroyAllWindows()
pca = RandomizedPCA(n_components=n_components, whiten=True)

param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),param_grid)

testing_data = []
for i in range(len(images)):
	testing_data.append(images[i].flatten())
pca = pca.fit(testing_data)

transformed = pca.transform(testing_data)
# if lda is done than #component = 80
#lda = LinearDiscriminantAnalysis(n_components=80)
#transformed = lda.fit(transformed, labels).transform(transformed)

clf.fit(transformed,labels)
directory2 = 'yalefaces_5' # test directory name
image_paths = [os.path.join(directory2, filename) for filename in os.listdir(directory2)]
j=0
for image_path in image_paths:
	pred_image_pil = Image.open(image_path).convert('L')
	pred_image = np.array(pred_image_pil, 'uint8')
	faces = faceCascade.detectMultiScale(pred_image)
	if(len(faces) ==0):
		continue
	if(faces.size != 4):
		continue;
	for (x,y,w,h) in faces:
		X_test = pca.transform(np.array(cv2.resize(pred_image[y:y+h,x:x+w] , dsize =(45,45))).flatten().reshape(1, -1))
#		X_test = lda.transform(X_test)
		mynbr = clf.predict(X_test)
		if os.path.split(image_path)[1][0:4] == 'male':
			nbr_act = 0;
		else:
			nbr_act = 1;
		print("Predicted By Classifier : ",mynbr[0], " Actual : ", nbr_act)
		if(mynbr[0] != nbr_act):
			j=j+1
			print(j)
print((400-j)/4)

"""

scale_factor=0.5
cap = cv2.VideoCapture(0)
#hand_cascade = cv2.CascadeClassifier('hand.xml')
success,frame=cap.read()
count=0
success=True
while True: 
    success,frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pred_image = np.array(gray, 'uint8')
    face = faceCascade.detectMultiScale(pred_image)
    cv2.imshow('video',gray)
    if len(face) == 0:
        print("Read a new frame:  Fail")
        continue
    else:
        pass
        #print("Read a new frame: ", success)

    for (x,y,w,h) in face:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),thickness=3)
        
        #crop_img = frame[y:y+h, x:x+w]
        sub_hand=frame[y:y+h,x:x+w]
        cv2.imwrite("frame.jpg", sub_hand)
        pred_image_pil = Image.open("frame.jpg").convert('L')
        pred_image = np.array(pred_image_pil, 'uint8')
        faces = faceCascade.detectMultiScale(pred_image)
        for (x,y,w,h) in faces:
            X_test = pca.transform(np.array(pred_image[y:y+col,x:x+row]).flatten().reshape(1, -1))
            mynbr = clf.predict(X_test)
            print(mynbr)
#      cv2.rectangle(frame,(x,y),(x+w-50,y+h+50),(0,255,0),thickness=4)
#      sub_hand=frame[y+h:y+h,x+w:(x+w)]
#      HandFileName = "unknownfaces_face" + str(y) + ".jpg"
#      cv2.imwrite(HandFileName, sub_hand)
#    cv2.imshow('video',frame)
#  if count%10==0:
#      cv2.imwrite("frame%d.jpg" % count, frame)      
    count += 1
    k=cv2.waitKey(10) & 0xff
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()

"""