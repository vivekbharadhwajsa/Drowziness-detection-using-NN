# Drowziness-detection-using-NN
This project deals with training a neural network to detect the state of the eyes and then using this data to detect whether the person is drowsy or not based model predictions on a webcam feed.

- The model is trained on MRL eye dataset which can be found at http://mrl.cs.vsb.cz/eyedataset

- The model was built on ResNet50 as backbone and trained using Keras with a **93.37 %** train accuracy

- The model weights is then used to predict the eye state in real-time on the images captured from the webcam using OpenCV

- **Face_detection_EAR.py** contains the drowziness detection algorithm based on Eye Aspect Ratio (EAR). This is based on the paper which can be found at https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf

Further extension can be the yawn detection along with the drowziness detection

**Face_recognition.ipynb** contains all the code put together. (faec_recog(5).ipynb) is the notebook downloaded after the training done on Google Colab.  

**results_new.txt** file contains model predictions along with the actual label put together as a tuple in the format (Model prediction, Actual label) for all the images in Test data.


**Softwares and tools**: **_Python, OpenCV, pandas, Keras, haar cascade and Jupyter Notebook_**
