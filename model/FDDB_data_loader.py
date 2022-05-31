import cv2
import os
import matplotlib.pyplot as plot
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
import numpy as np

class FDDB_Data:
    def __init__(self, Face_image_dir,NonFace_image_dir,face_test_dir, NonFace_Test_image_dir):
        self.Face_image_dir = Face_image_dir
        self.NonFace_image_dir = NonFace_image_dir
        self.face_test_dir = face_test_dir
        self.NonFace_Test_image_dir = NonFace_Test_image_dir

    def load(self, train):
        if train:
            Face_images = os.listdir(self.Face_image_dir)
            NonFace_images = os.listdir(self.NonFace_image_dir)

            Face_Vector_Space = []
            for face_image in Face_images:
                image = cv2.imread(self.Face_image_dir+face_image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (10,10))
                im_vector = image.flatten()
                Face_Vector_Space.append(im_vector)
            Face_Vector_Space = np.array(Face_Vector_Space)

            NonFace_vector_space = []
            for NonFace_image in NonFace_images:
                image = cv2.imread(self.NonFace_image_dir+NonFace_image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (10,10))
                im_vector = image.flatten()
                NonFace_vector_space.append(im_vector)
            NonFace_vector_space = np.array(NonFace_vector_space)

            return Face_Vector_Space, NonFace_vector_space
        else:
            Face_test_images = os.listdir(self.Face_Test_image_dir) #write the directory that has the text file
            NonFace_test_images = os.listdir(self.NonFace_Test_image_dir)
            Face_test_vector_space = []
            for Face_test_image in Face_test_images:
                image = cv2.imread(self.Face_Test_image_dir+Face_test_image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (10,10))
                im_vector = image.flatten()
                Face_test_vector_space.append(im_vector)
            Face_test_vector_space = np.array(Face_test_vector_space)
            #print("Done for Face test",Face_test_vector_space.shape)
            
            NonFace_test_vector_space = []
            for NonFace_test_image in NonFace_test_images:
                #print(NonFace_test_image)

                image = cv2.imread(self.NonFace_Test_image_dir+NonFace_test_image) 
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (10,10))
                im_vector = image.flatten()
                NonFace_test_vector_space.append(im_vector)
            NonFace_test_vector_space = np.array(NonFace_test_vector_space)

            return Face_test_vector_space, NonFace_test_vector_space
