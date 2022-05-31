# This script is used for creating the dataset for training our model. 
# We crop the face and non-face images and store them inside the "negImages_gray" and "posImages_gray" folder
from re import T
import cv2
import linecache
import random
import numpy as np
#Assigning the directory from which images are to be read
root_dir_ImageDetails = "/home/hshukla/Env/ComputerVision/ECE763_Project1/Data/FDDB-folds/"
root_dir_originalImages = "/home/hshukla/Env/ComputerVision/ECE763_Project1/Data/originalPics/"
root_dir_FaceImages = "/home/hshukla/Env/ComputerVision/ECE763_Project1/Data/FaceImages_gray/"
root_dir_NonFaceImages = "/home/hshukla/Env/ComputerVision/ECE763_Project1/Data/Non_FaceImages_gray/"
im_format = ".jpg"

#Initializing variables
folds = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
count = 1
neg_image_size = 60

#Initial values for 
NonFace_x1=0
NonFace_x2=0
NonFace_y1=0
NonFace_y2=0

# For loop for extracting images from all 10 folds as given in the text files
for num_in_fddb_fold in folds:
    filename_ellipse = root_dir_ImageDetails+"FDDB-fold-"+num_in_fddb_fold+"-ellipseList.txt"
    filename_images = root_dir_ImageDetails+"FDDB-fold-"+num_in_fddb_fold+".txt"
    file_images = open(filename_images,'r') #Opening FDDB-fold-<>.txt -- Contains name of images 
    image_names = file_images.readlines() #returns list of lines -- List of image names 
    #Count number of lines to get the number of images in each numb
    number_of_images = len(image_names)
    for image_name in image_names:
        with open(filename_ellipse,'r') as file_ellipse:  #Contains the dimension of ellipses for each faces
            for num, line in enumerate(file_ellipse, 1):
                if image_name in line:
                    num_face_line = linecache.getline(filename_ellipse, (num+1))
                    if num_face_line.strip() == "1":
                        ellipse_dim = (linecache.getline(filename_ellipse, (num+2))).split()
                        image = cv2.imread(root_dir_originalImages+image_name.strip()+im_format)
                        #Get the height and width of the image
                        height,width, channel = image.shape
                        #Create bounding box coordinates for Face images
                        Face_x1 = int(float(ellipse_dim[3])-float(ellipse_dim[1]))         
                        Face_x2 = int(float(ellipse_dim[3])+float(ellipse_dim[1]))            
                        Face_y1 = int(float(ellipse_dim[4])-float(ellipse_dim[0]))           
                        Face_y2 = int(float(ellipse_dim[4])+float(ellipse_dim[0]))            
                        height_face,width_face = (Face_y2-Face_y1),(Face_x2-Face_x1)
                        #Get the Face image by cropping from the original image
                        cropped_Face_image = image[Face_y1:Face_y2,Face_x1:Face_x2]
                        #Create a file of this cropped face image in the "posImages_gray" folder
                        if (cropped_Face_image.shape[0]) and (cropped_Face_image.shape[1]):
                            count = count+1
                            cropped_Face_image = cv2.resize(cropped_Face_image, (60,60))
                            gray_im = cv2.cvtColor(cropped_Face_image, cv2.COLOR_RGB2GRAY )
                            cv2.imwrite(root_dir_FaceImages+str(count)+im_format, gray_im)
                        
                        #Define a array for stroing non-face image cropped from the background of the same image from which the face is cropped
                        cropped_NonFace_image = np.ones((cropped_Face_image.shape[0],cropped_Face_image.shape[1],3))
                        
                        # CREATING Non-face data set by cropping a random background from the image
                        # 4 if conditions are used to check space for cropping a non-face image in all 4 directions of the detected face  
                        # Search for non-face towards the left and down of the face detected in the image  <Condition 1>
                        if ((Face_y1>height_face+10) and (Face_x1>width_face+10)):
                            cropped_NonFace_image = image[0:height_face,0:Face_x1]
                            NonFace_y1 = 0
                            NonFace_y2 = height_face-1
                            NonFace_x1 = 0
                            NonFace_x2 = Face_x1-1
                           
                        # Search for non-face towards the left and up of the face detected in the image  <Condition 2>
                        elif (((height-Face_y2)>height_face+10) and (Face_x1>width_face+10)):
                            cropped_NonFace_image = image[Face_y2+10:Face_y2+10+height_face,0:Face_x1]
                            NonFace_y1 = Face_y2+10
                            NonFace_y2 = Face_y2+10+height_face-1
                            NonFace_x1 = 0
                            NonFace_x2 = Face_x1-1

                        # Search for non-face towards the right and down of the face detected in the image  <Condition 3>  
                        elif ((Face_y1>height_face+10) and (Face_x1<width_face+10)):
                            cropped_NonFace_image = image[0:height_face,Face_x1+10:Face_x1+10+width_face]
                            NonFace_y1 = 0
                            NonFace_y2 = height_face-1
                            NonFace_x1 = Face_x1+10
                            NonFace_x2 = Face_x1+10+width_face-1
                         
                        # Search for non-face towards the right and up of the face detected in the image  <Condition 4>
                        elif (((height-Face_y2)>height_face+10) and (Face_x1<width_face+10)):
                            cropped_NonFace_image = image[Face_y2+10:Face_y2+10+height_face,Face_x1+10:Face_x1+10+width_face]
                            NonFace_y1 = Face_y2+10
                            NonFace_y2 = Face_y2+10+height_face-1
                            NonFace_x1 = 0
                            NonFace_x2 = Face_x1+10+width_face-1
                            #print("Condition 4")
                        else:
                            #If there is no space available in the original image to crop any non-face image, then we create a random non-face image as follows 
                            #Generate random number from 0 to 255 in t
                            for i in range (0,cropped_NonFace_image.shape[0]):
                                for j in range (0,cropped_NonFace_image.shape[1]):
                                    cropped_NonFace_image[i,j] = random.randint(0,255)
                       
                        if (cropped_NonFace_image.shape[0]) and (cropped_NonFace_image.shape[1]):
                            count = count+1
                            cropped_NonFace_image = cv2.cvtColor(np.float32(cropped_NonFace_image), cv2.COLOR_RGB2GRAY)
                            cropped_NonFace_image = cv2.resize(cropped_NonFace_image, (60, 60))                        
                            cv2.imwrite(root_dir_NonFaceImages+str(count)+im_format, cropped_NonFace_image)
                            
