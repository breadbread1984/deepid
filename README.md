# Tutorial on deepid
### Introduction

This code is an implement of the face recognition algorithm introduced in paper [Deep Learning Face Representation from Predicting 10,000 Classes](http://mmlab.ie.cuhk.edu.hk/pdf/YiSun_CVPR14.pdf).

### Project Structure

train_model:  training models and solvers for deepid feature extractor and face recognizer

deploy_model: deploy models for deepid feature extractor and face recognizer

model_values: trained model files

src: source codes

### Building

```Shell
make -j9
```

### Training Feature Extractor

You can skip the training if you just want to detect facial landmarkers with this project because all pretrained deepid caffemodel files are given in model values directory
1. request [CASIA-webface](www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) dataset.

2. crop patches from datasets as described in the paper

in root directory

```Shell
./generate_training_sample -i <path/to/webface> -o <path/to/training patches for 60 models>
```
3. convert patch samples into LMDB files

in root directory

```Shell
./convert -i <path/to/training patches for 60 models> -o <path/to/lmdbs>
```

4. train models

enter each generated lmdb directory for patches of one of 60 local facial areas and train with caffe.

```Shell
caffe train -solver deepid_solver.prototxt -gpu all
```

5. collect trained model files

in root directory

```Shell
./move_training_results -i <path/to/lmdbs> -o model_values
```

### Training Face Recognizer

1. extract deepid features from webface faces

in root directory

```Shell
./generate_training_samples  -i  <path/to/webface>  -o <path/to/lmdb of deepid feature>
```

2. train face recognizer

train with caffe on the generated lmdb of deepid feature. My deepid code can only achieve 70% accuracy, trained and tested on webface.

### Training Face Recognizer on dimension reduced (150d) feature vectores

1. calculate principal vectors of deepid features and transform them into principal component subspaces

```Shell
./transform  -i  <path/to/lmdb of deepid feature>  -o  <path/to/lmdb of dimension reduced deepid features>
```

This operation takes me around 3 weeks on my workstation to finish.

2. train face recognizer

train with caffe on the generated lmdb of dimension reduced deepid feature. The accuracy is even lower, only 65% accuracy.

### Demo of Face Recognition with deepid feature extractor + knn

1. collect face pictures to recognizer

put (at least one) picture(s) for each person into individual directories and put all the directories into one parent directory.

2.  extract deepid features

in root directory

extract deepid feaures from faces

```Shell
./main -m train -i <path/to/the parent directory>
```

3. play with your webcam

recognize with knn from pictures captured from webcam

```Shell
./main -m test -p  训练参数.dat
```
