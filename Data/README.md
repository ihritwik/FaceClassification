### Dataset Description
Data set source: FDDB: Face Detection Data Set and Benchmark.  

Set Contains:
1. Original set of images (from Faces in the Wild data set)
2. Face annotations
  * Uncompressing the **"FDDB-folds.tgz"** file creates a directory **"FDDB-folds"**,
which contains files with names: **FDDB-fold-xx.txt** and
**FDDB-fold-xx-ellipseList.txt**, where xx = {01, 02, ..., 10} represents the
fold-index.
  * Each line in the **"FDDB-fold-xx.txt"** file specifies a path to an image in the
above-mentioned data set. For instance, the entry **"2002/07/19/big/img_130"**
corresponds to **"originalPics/2002/07/19/big/img_130.jpg"**
  * The corresponding annotations are included in the file
**"FDDB-fold-xx-ellipseList.txt"** in the following format:  
..  
{image name i}  
{number of faces in this image = im}  
{face i1}  
{face i2}  
..  
..  
{face im}  
..  

Here, each face is denoted by:
{major_axis_radius minor_axis_radius angle center_x center_y 1}.

[Link]: http://vis-www.cs.umass.edu/fddb/
