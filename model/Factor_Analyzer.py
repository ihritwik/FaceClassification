import matplotlib.pyplot as plot
from sklearn.metrics import roc_curve, auc
from FDDB_data_loader import FDDB_Data
import cv2
import os
from random import randint
import numpy as np
def train_and_test_factor_analyzer(Face_vector_space, NonFace_vector_space, FaceTestVectorSpace, NonFaceTestVectorSpace,
                              args.save_dir):
    def FactorAnalyzerPDF(x, mean, covar, phi):
        difference = x - mean
        Temp1 = np.matmul(phi, np.ndarray.transpose(phi))
        Temp2 = np.matmul(difference[None,:],np.linalg.inv(covar + Temp1))
        Temp3 = (np.matmul(Temp2,difference[:,None]))
        prob_distribution = np.exp((-0.5)*Temp3)
        return prob_distribution

    # fddb_data = FDDB_Data()
    # Face_vector_space = fddb_data.load(train=True)[0]
    # NonFace_vector_space = fddb_data.load(train=True)[1]

    num_factors = 5                                 
    num_iterations = 10                             

    Face_basis_vectors = np.random.random_sample((Face_vector_space.shape[1], num_factors)) + 1
    NonFace_basis_vectors = np.random.random_sample((Face_vector_space.shape[1], num_factors)) + 1
    Face_mean_vector = np.mean(Face_vector_space, axis = 0)
    NonFace_mean_vector = np.mean(NonFace_vector_space, axis = 0)
    Face_covariance = np.cov((Face_vector_space), rowvar = False)
    NonFace_covariance = np.cov((NonFace_vector_space), rowvar = False)

    Face_fa = np.zeros((Face_vector_space.shape[0],num_factors))
    Face_fa_ra = np.zeros((num_factors,num_factors,Face_vector_space.shape[0]))
    NonFace_fa = np.zeros((NonFace_vector_space.shape[0],num_factors))
    NonFace_fa_ra = np.zeros((num_factors,num_factors,NonFace_vector_space.shape[0]))

    # E-M Algorithm
    for iter in range(num_iterations):
        print("Iteration #: ", iter+1)

        # E-step
        print("Expectation Step in EM Algorithm")
        for i in range(Face_vector_space.shape[0]):
            Prod_1 = np.matmul(Face_basis_vectors.transpose(), np.linalg.inv(Face_covariance))
            Prod_2 = np.matmul(Prod_1, Face_basis_vectors) + np.eye(num_factors)
            term1 = np.linalg.inv(Prod_2 + np.diag(np.diag(np.ones(num_factors))))
            Prod_3 = np.matmul(term1, Face_basis_vectors.transpose())
            Prod_4 = np.matmul(Prod_3, np.linalg.inv(Face_covariance))
            difference = Face_vector_space[i] - Face_mean_vector
            Face_fa[i,:] = np.matmul(Prod_4, difference)
            Face_fa_ra[:,:,i] = term1 + np.matmul(Face_fa[i,:], Face_fa[i,:].transpose())

        for i in range(NonFace_vector_space.shape[0]):
            mul1 = np.matmul(NonFace_basis_vectors.transpose(), np.linalg.inv(NonFace_covariance))
            Prod_2 = np.matmul(Prod_1, NonFace_basis_vectors) + np.eye(num_factors)
            term1 = np.linalg.inv(Prod_2 + np.diag(np.diag(np.ones(num_factors))))
            Prod_3 = np.matmul(term1, NonFace_basis_vectors.transpose())
            Prod_4 = np.matmul(Prod_3, np.linalg.inv(NonFace_covariance))
            difference = NonFace_vector_space[i] - NonFace_mean_vector
            NonFace_fa[i,:] = np.matmul(Prod_4, difference)
            NonFace_fa_ra[:,:,i] = term1 + np.matmul(NonFace_fa[i,:], NonFace_fa[i,:].transpose())

        # M-step
        print("Maximization Step in EM ALgorithm")
        Face_mean_vector = np.sum(Face_vector_space, axis=0)/Face_vector_space.shape[0]
        NonFace_mean_vector = np.sum(NonFace_vector_space, axis=0)/NonFace_vector_space.shape[0]

        for i in range(Face_vector_space.shape[0]):
            difference = Face_vector_space[i] - Face_mean_vector
            term1 = np.matmul(difference[:,None], Face_fa[i,:,None].transpose())

        Face_basis_vectors = np.matmul(term1, np.linalg.inv(np.sum(Face_fa_ra, axis=2)))

        temp_covar_num = np.zeros((Face_vector_space.shape[1], Face_vector_space.shape[1]))
        for i in range(Face_vector_space.shape[0]):
            difference = Face_vector_space[i] - Face_mean_vector
            Prod_1 = np.matmul(difference[:,None], difference[None,:])
            Prod_2 = np.matmul(Face_basis_vectors, Face_fa[i,:,None])
            Prod_3 = np.matmul(Prod_2,difference[None,:])
            temp_covar_num = temp_covar_num + np.diag(np.diag(Prod_1 - Prod_3))
        Face_covariance = temp_covar_num/Face_vector_space.shape[0]

        for i in range(NonFace_vector_space.shape[0]):
            difference = NonFace_vector_space[i] - NonFace_mean_vector
            term1 = np.matmul(difference[:,None], NonFace_fa[i,:,None].transpose())

        NonFace_basis_vectors = np.matmul(term1, np.linalg.inv(np.sum(NonFace_fa_ra, axis=2)))

        temp_covar_num = np.zeros((NonFace_vector_space.shape[1], NonFace_vector_space.shape[1]))
        for i in range(NonFace_vector_space.shape[0]):
            difference = NonFace_vector_space[i] - NonFace_mean_vector
            mul1 = np.matmul(difference[:,None], difference[None,:])
            Prod_2 = np.matmul(NonFace_basis_vectors, NonFace_fa[i,:,None])
            Prod_3 = np.matmul(Prod_2,difference[None,:])
            temp_covar_num = temp_covar_num + np.diag(np.diag(Prod_1 - Prod_3))
        NonFace_covariance = temp_covar_num/NonFace_vector_space.shape[0]

    FacePosterior_P = np.zeros(FaceTestVectorSpace.shape[0])
    FacePosterior_N = np.zeros(FaceTestVectorSpace.shape[0])
    NonFacePosterior_P = np.zeros(NonFaceTestVectorSpace.shape[0])
    NonFacePosterior_N = np.zeros(NonFaceTestVectorSpace.shape[0])

    for i in range(FaceTestVectorSpace.shape[0]):
        Facelikelihood_P = FactorAnalyzerPDF(FaceTestVectorSpace[i], Face_mean_vector, Face_covariance, Face_basis_vectors)
        Facelikelihood_N = FactorAnalyzerPDF(NonFaceTestVectorSpace[i], Face_mean_vector, Face_covariance, Face_basis_vectors)
        NonFacelikelihood_P = FactorAnalyzerPDF(FaceTestVectorSpace[i], NonFace_mean_vector, NonFace_covariance, NonFace_basis_vectors)
        NonFacelikelihood_N = FactorAnalyzerPDF(NonFaceTestVectorSpace[i], NonFace_mean_vector, NonFace_covariance, NonFace_basis_vectors)

        FacePosterior_P[i] = Facelikelihood_P/(Facelikelihood_P+NonFacelikelihood_P)
        FacePosterior_N[i] = Facelikelihood_N/(Facelikelihood_N+NonFacelikelihood_N)
        NonFacePosterior_P[i] = NonFacelikelihood_P/(Facelikelihood_P+NonFacelikelihood_P)
        NonFacePosterior_N[i] = NonFacelikelihood_N/(Facelikelihood_N+NonFacelikelihood_N)

    Posterior = np.append(FacePosterior_P, FacePosterior_N)
    labels = np.append(np.ones(len(FacePosterior_P)), np.zeros(len(FacePosterior_N))   )

    cv2.imwrite("Face_Mean_FactorAnalyzer.png", Face_mean_vector.reshape((10,10)).astype('uint8'))
    cv2.imwrite("NonFace_Mean_FactorAnalyzer.png", NonFace_mean_vector.reshape((10,10)).astype('uint8'))
    cv2.imwrite("Face_Covariance_FactorAnalyzer.png", Face_covariance)
    cv2.imwrite("NonFace_Covariance_FactorAnalyzer.png", NonFace_covariance)

    fpr, tpr, _ = roc_curve(labels, Posterior, pos_label=1)
    plot.plot(fpr, tpr, color='darkorange')
    print("False Positive Rate: {}".format(fpr[int(fpr.shape[0]/2)]))
    print("False Negative Rate: {}".format(1-tpr[int(fpr.shape[0]/2)]))
    print("Misclassification Rate: {}".format(fpr[int(fpr.shape[0]/2)] + (1-tpr[int(fpr.shape[0]/2)])))
    plot.xlim([-0.1,1.1])
    plot.ylim([-0.1,1.1])
    plot.title("ROC for Factor Analyzer Classifier")
    plot.ylabel("True Positives")
    plot.xlabel("False Positives")
    plot.show()
