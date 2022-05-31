import numpy as np
import cv2
import os
from random import randint
import matplotlib.pyplot as plot
from sklearn.metrics import roc_curve, auc
from FDDB_data_loader import FDDB_Data
def normpdf(x, mean, covar):
    diff = x - mean
    dell1 = np.matmul(diff[None,:],np.linalg.inv(covar))
    dell = (np.matmul(dell1,diff[:,None]))
    num = np.exp((-0.5)*dell)
    return num

fddb_data = FDDB_Data()
Face_Vector_Space = fddb_data.load(train=True)[0]
NonFace_Vector_Space = fddb_data.load(train=True)[1]

num_mixtures = 5                               
num_iterations = 10                              

Face_mix_weights = np.zeros(num_mixtures)
for i in range(num_mixtures):
    Face_mix_weights[i] = 1/num_mixtures

# Init negative mixture weights
NonFace_mix_weights = np.zeros(num_mixtures)
for i in range(num_mixtures):
    NonFace_mix_weights[i] = 1/float(num_mixtures)

Face_mean_vector_space = np.random.randint(0, 255, (num_mixtures, Face_Vector_Space.shape[1]))

# Init Mean for negative images
NonFace_mean_vector_space = np.random.randint(0, 255, (num_mixtures, NonFace_Vector_Space.shape[1]))

# Init Positive Covariance matrix
Face_co_var_space = np.empty((100,100,num_mixtures))
for i in range(num_mixtures):
    Face_co_var_space[:,:,i] = (i+1)*(np.eye(100))
Face_co_var_space = 5000*Face_co_var_space

# Init Negative Covariance matrix
NonFace_co_var_space = np.empty((100,100,num_mixtures))
for i in range(num_mixtures):
    NonFace_co_var_space[:,:,i] = (i+1)*(np.eye(100))
NonFace_co_var_space = 5000*NonFace_co_var_space

pos_responsibility = np.zeros((len(Face_Vector_Space),num_mixtures))
neg_responsibility = np.zeros((len(NonFace_Vector_Space),num_mixtures))

for iter in range(num_iterations):
    print("Current Iteration : ", iter+1)
    for k in range(num_mixtures):
        Face_co_var_space[:,:,k] = (np.diag(np.diag(Face_co_var_space[:,:,k])+1))
        NonFace_co_var_space[:,:,k] = (np.diag(np.diag(NonFace_co_var_space[:,:,k])+1))

    # E-step
    print("Running Expectation Step of EM Algorithm")
    for k in range(num_mixtures):
        #print("Cluster #: ",k+1)
        for i in range(Face_Vector_Space.shape[0]):

            pos_likelihood = normpdf(Face_Vector_Space[i,:], Face_mean_vector_space[k,:], Face_co_var_space[:,:,k])
            neg_likelihood = normpdf(NonFace_Vector_Space[i,:], NonFace_mean_vector_space[k,:], NonFace_co_var_space[:,:,k])

            pos_evidence = 0
            neg_evidence = 0
            for j in range(num_mixtures):
                pos_evidence = pos_evidence + Face_mix_weights[j]*normpdf(Face_Vector_Space[i,:], Face_mean_vector_space[j,:], Face_co_var_space[:,:,j])
                neg_evidence = neg_evidence + NonFace_mix_weights[j]*normpdf(NonFace_Vector_Space[i,:], NonFace_mean_vector_space[j,:], NonFace_co_var_space[:,:,j])

            pos_responsibility[i,k] = (Face_mix_weights[k]*pos_likelihood)/pos_evidence
            neg_responsibility[i,k] = (NonFace_mix_weights[k]*neg_likelihood)/neg_evidence

    # M-step:
    print("Running Maximization step of EM Algorithm")
    Face_mix_weights = np.sum(pos_responsibility, axis = 0)/np.sum(np.sum(pos_responsibility, axis = 0))
    neg_mixt_weights = np.sum(neg_responsibility, axis = 0)/np.sum(np.sum(neg_responsibility, axis = 0))
    print("Updated Face Weights after iteration {}: ".format(iter+1), Face_mix_weights)
    print("Updated NonFace Weights after iteration {}: ".format(iter+1), NonFace_mix_weights)
    for k in range(num_mixtures):
        print("Cluster #: ", k+1)
        num = np.zeros(Face_Vector_Space.shape[1])
        for i in range(len(Face_Vector_Space)):
            num = num + pos_responsibility[i,k]*(Face_Vector_Space[i,:])
        Face_mean_vector_space[k,:] = num/np.sum(pos_responsibility[:,k])

        num = np.zeros(NonFace_Vector_Space.shape[1])
        for i in range(len(NonFace_Vector_Space)):
            num = num + neg_responsibility[i,k]*(NonFace_Vector_Space[i,:])
        NonFace_mean_vector_space[k,:] = num/np.sum(neg_responsibility[:,k])

        numer = np.zeros((Face_Vector_Space.shape[1], Face_Vector_Space.shape[1]))
        for i in range(len(Face_Vector_Space)):
            diff = (Face_Vector_Space[i,:] - Face_mean_vector_space[k,:])
            dell = np.matmul(diff[:,None],diff[None,:])
            numer = numer + pos_responsibility[i,k]*dell
        Face_co_var_space[:,:,k] = numer/np.sum(pos_responsibility[:,k])

        numer = np.zeros((NonFace_Vector_Space.shape[1], NonFace_Vector_Space.shape[1]))
        for i in range(len(NonFace_Vector_Space)):
            diff = (NonFace_Vector_Space[i,:] - NonFace_mean_vector_space[k,:])
            dell = np.matmul(diff[None,:],diff[:,None])
            numer = numer + neg_responsibility[i,k]*dell
        NonFace_co_var_space[:,:,k] = numer/np.sum(neg_responsibility[:,k])

# cv2.imshow("Mean Face", pos_mean_vector_space.reshape((10,10)).astype('uint8'))
for i in range(num_mixtures):
    cv2.imwrite("Face_Mean_MixtureGaussian"+str(i+1)+".png", Face_mean_vector_space[i].reshape((10,10)).astype('uint8'))
    cv2.imwrite("NonFace_Mean_MixtureGaussian"+str(i+1)+".png", NonFace_mean_vector_space[i].reshape((10,10)).astype('uint8'))
    cv2.imwrite("Face_Covariance_GaussianMixture"+str(i+1)+".png", Face_co_var_space[:,:,i])
    cv2.imwrite("NonFace_Covariance_GaussianMixture"+str(i+1)+".png", NonFace_co_var_space[:,:,i])
    print("All images saved !!")
for k in range(num_mixtures):
    Face_co_var_space[:,:,k] = (np.diag(np.diag(Face_co_var_space[:,:,k])+1))
    NonFace_co_var_space[:,:,k] = (np.diag(np.diag(NonFace_co_var_space[:,:,k])+1))

# Load testing samples
pos_test_vector_space = fddb_data.load(train=False)[0]
neg_test_vector_space = fddb_data.load(train=False)[1]

FacePosterior_P = np.zeros(pos_test_vector_space.shape[0])
FacePosterior_N = np.zeros(pos_test_vector_space.shape[0])
NonFacePosterior_P = np.zeros(neg_test_vector_space.shape[0])
NonFacePosterior_N = np.zeros(neg_test_vector_space.shape[0])
for i in range(pos_test_vector_space.shape[0]):
    Facelikelihood_P = 0
    Facelikelihood_N = 0
    NonFacelikelihood_P = 0
    NonFacelikelihood_N = 0
    for k in range(num_mixtures):
        Facelikelihood_P = Facelikelihood_P + Face_mix_weights[k]*normpdf(pos_test_vector_space[i], Face_mean_vector_space[k,:], Face_co_var_space[:,:,k])
        Facelikelihood_N = Facelikelihood_N + Face_mix_weights[k]*normpdf(neg_test_vector_space[i], Face_mean_vector_space[k,:], Face_co_var_space[:,:,k])
        NonFacelikelihood_P = NonFacelikelihood_P + NonFace_mix_weights[k]*normpdf(pos_test_vector_space[i], NonFace_mean_vector_space[k,:], NonFace_co_var_space[:,:,k])
        nNonFacelikelihood_N = nNonFacelikelihood_N + NonFace_mix_weights[k]*normpdf(neg_test_vector_space[i], NonFace_mean_vector_space[k,:], NonFace_co_var_space[:,:,k])

    FacePosterior_P[i] = Facelikelihood_P/(Facelikelihood_P+NonFacelikelihood_P)
    FacePosterior_N[i] = Facelikelihood_N/(Facelikelihood_N+NonFacelikelihood_N)
    NonFacePosterior_P[i] = NonFacelikelihood_P/(Facelikelihood_P+NonFacelikelihood_P)
    NonFacePosterior_N[i] = NonFacelikelihood_N/(Facelikelihood_N+NonFacelikelihood_N)

Posterior = np.append(FacePosterior_P, FacePosterior_N)
labels = np.append(np.ones(len(FacePosterior_P)), np.zeros(len(FacePosterior_P))   )

fpr, tpr, _ = roc_curve(labels, Posterior, pos_label=1)
print("FPR = ")
print (fpr)
print("TOR = ")
print (tpr)
plot.plot(fpr, tpr, color='blue')
print("False Positive Rate: {}".format(fpr[int(fpr.shape[0]/2)]))
print("False Negative Rate: {}".format(1-tpr[int(fpr.shape[0]/2)]))
print("Misclassification Rate: {}".format(fpr[int(fpr.shape[0]/2)] + (1-tpr[int(fpr.shape[0]/2)])))
plot.xlim([-0.1,1.1])
plot.ylim([-0.1,1.1])
plot.title("ROC for Gaussian Mixture Classifier")
plot.ylabel("True Positives")
plot.xlabel("False Positives")
plot.show()
