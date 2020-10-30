# Drowziness-detection-using-NN
This project deals with training a neural network to detect the state of the eyes and then using this data to detect whether the person is drowsy or not based model predictions on a webcam feed.

- The model is trained on MRL eye dataset which can be found at http://mrl.cs.vsb.cz/eyedataset

- The model was built on ResNet50 as backbone and trained using Keras with a 93% train accuracy

- The model weights is then used to predict the eye state in real-time on the images captured from the webcam using OpenCV

- Face_detection_EAR.py contains the drowziness detection algorithm based on Eye Aspect Ratio (EAR). This is based on the paper which can be found at https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf

Further extension can be the yawn detection along with the drowziness detection


**Softwares: Python, OpenCV, pandas, Keras, haar cascade and Jupyter Notebook**
