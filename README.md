# Skin Carcinoma Detection Using Parallel Deep Residual Networks
A deep neural network developed following the residual learning and separable convolution paradigms to diagnose basal and squamous cell carcinoma using a subset of ISIC dataset. The network has been designed using the Keras framework.

# Data
The dataset used for this project can be found within the "dataset" folder. This dataset is a subset of the ISIC archive data.

# Data preprocessing
The image data has already been resized to 224 x 224 dimensions. However, we apply further preprocessing steps which can be implemented using the "preprocessing.py" file.

# Model
The deep neural network devised by us is defined in the "carcinomaNetwork.py" file. Please use the compile_model() method to obtain the compiled model.

# Training
To train the model use the "training.py" file. We found the best kernel sizes to be (1 x 4) and (4 x 1) for the subnetworks. To change the optimizer for the model, refer to the compile_model() method in the "carcinomaNetwork.py" file. Feel free to include/modify necessary callbacks and related training parameters.

# Performance scores
To obtain the performance scores of the model one may use our "performance_viz.py" file, which includes methods to visualiza the confusion matrix. One may use keras-viz to visualize the grad-CAMs for the convolution layers. We also provide a method to plot the ROC curves for the model.
