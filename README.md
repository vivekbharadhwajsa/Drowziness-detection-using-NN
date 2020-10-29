# Drowziness-detection-using-NN
This project deals with training a neural network to detect the state of the eyes and then using this data to detect whether the person is drowsy or not based model prediction on a webcam feed.

The model is trained on MRL eye dataset which can be found at http://mrl.cs.vsb.cz/eyedataset

The model was built on ResNet50 as backbone and trained using Keras with a 93% train accuracy

The model weights is then used to predict the eye state in real-time on the images captured from the webcam using OpenCV

Further extension can be the yawn detection along with the drowziness detection
(This project does not use the Eye Aspect Ratio(EAR) in order to detect drowziness. The file extension for drowziness detection using EAR will be added later)


**Softwares: Python, OpenCV, pandas, Keras, haar cascade and Jupyter Notebook**
