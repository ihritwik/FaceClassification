import os
import sys
import numpy as np

curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(curr_path, '../'))

from model.Factor_Analyzer import *
from model.Single_Gaussian import *
from model.T_Distribution import *
from model.Gaussian_Mixture import *
from model.FDDB_data_loader import FDDB_Data

def parse_args():
    parser = argparse.ArgumentParser(
        description='Crop Face and Non-Face Patcehs from FDDB')
    # general
    parser.add_argument(
        '--Face_image_dir', help='path storing training images of faces', required=True, type=str)
    parser.add_argument(
        '--NonFace_image_dir', help='path storing training images of Non-faces', required=True, type=str)
    parser.add_argument(
        '--Face_Test_image_dir', help='path storing testing images of faces', required=True, type=str)
    parser.add_argument(
        '--NonFace_Test_image_dir', help='path storing testing images of non-faces', default=True, type=bool)
    
    args = parser.parse_args()
    if args.shuffle:
        args.ignore_existed = True

    return args

def main():
    args = parse_args()

    # training dataset
    # FDDB dataset is avaiable at
    #   images: http://vis-www.cs.umass.edu/fddb/originalPics.tar.gz
    #   annotations: http://vis-www.cs.umass.edu/fddb/FDDB-folds.tgz
    
    fddb_data = FDDB_Data(args.Face_image_dir,args.NonFace_image_dir,args.Face_Test_image_dir,args.NonFace_Test_image_dir)
    Face_Vector_Space = fddb_data.load(train=True)[0]
    NonFace_Vector_Space = fddb_data.load(train=True)[1]
    
    # testing dataset
    FaceTestVectorSpace = fddb_data.load(train=False)[0]
    NonFaceTestVectorSpace = fddb_data.load(train=False)[1]
    
    # Single Gaussian Model
    train_and_test_norm(Face_Vector_Space, NonFace_Vector_Space, FaceTestVectorSpace, NonFaceTestVectorSpace)

    # MoG
    loglikelihood_threshold = 0.1
    maxiter = 100
    minK = 2
    maxK = 11

    train_and_test_MoG(Face_Vector_Space, NonFace_Vector_Space, FaceTestVectorSpace, NonFaceTestVectorSpace,
                       minK, maxK, maxiter, loglikelihood_threshold)


    # Student's t-distri
    train_and_test_t_distr(Face_Vector_Space, NonFace_Vector_Space, FaceTestVectorSpace, NonFaceTestVectorSpace,
                           args.diag_cov, args.save_dir)


    # [Optional] Mixture of t-distrs.
    # your code here


    # Factor analyzer
    num_factors = 5                                 
    num_iterations = 10 
    
    train_and_test_factor_analyzer(Face_Vector_Space, NonFace_Vector_Space, FaceTestVectorSpace, NonFaceTestVectorSpace,
                              num_factors, num_iterations)


    # [Optional] Mixture of FA.
    # your code here


if __name__ == '__main__':
    main()
