####Importing the libraries####
from keras import backend as K
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import SeparableConv2D, BatchNormalization, MaxPooling2D
from keras.layers import Input, Add, Concatenate, ELU, ReLU
from keras.initializers import VarianceScaling
from keras.models import Model
from keras.optimizers import Adam, RMSprop, SGD

####Desiging the residual block####
def residual_block(model, f_in, f_out, k):
    
    shortcut = model
    
    model = SeparableConv2D(f_in, kernel_size = k, strides = (1,1), padding = "same",
                            kernel_initializer = VarianceScaling(scale=1, mode = "fan_in", distribution = "normal", seed = None),
                            bias_initializer = "zeros")(model)
    model = BatchNormalization()(model)
    model = ELU()(model)
    
    model = SeparableConv2D(f_out, kernel_size = k, strides = (1,1), padding = "same",
                            kernel_initializer = VarianceScaling(scale=1, mode = "fan_in", distribution = "normal", seed = None),
                            bias_initializer = "zeros")(model)
    model = BatchNormalization()(model)
    
    if f_in != f_out:
        shortcut = SeparableConv2D(f_out, kernel_size = k, strides = (1,1), padding = "same",
                                   kernel_initializer = VarianceScaling(scale=1, mode = "fan_in", distribution = "normal", seed = None),
                                   bias_initializer = "zeros")(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    model = Add()([model, shortcut])
    model = ReLU()(model)
    return model
	
####Designing individual sub networks####
def branch_(mod_, k):
    
    model = mod_
    
    model = residual_block(model, 16, 32, k)
    model = MaxPooling2D()(model)
    
    model = residual_block(model, 32, 48, k)
    model = MaxPooling2D()(model)
    
    model = residual_block(model, 48, 64, k)
    model = MaxPooling2D()(model)
    
    model = residual_block(model, 64, 96, k)
    model = MaxPooling2D()(model)
    
    model = GlobalAveragePooling2D()(model)
    
    return model
	
####Designing the classification network####
def model_build(in_, k):
    
    model = SeparableConv2D(8, kernel_size = (1,1), strides = (1,1), padding = "same",
                            kernel_initializer = VarianceScaling(scale = 1, mode = "fan_in", distribution = "normal", seed = None), 
                            bias_initializer = "zeros")(in_)
    model = BatchNormalization()(model)
    model = ReLU()(model)
    
    horizontal_branch = branch_(model, (1,k))
    vertical_branch = branch_(model, (k,1))
    
    model = Concatenate(axis = -1)([horizontal_branch, vertical_branch])

    model = Dense(64, activation = "relu")(model)
    model = Dropout(0.5)(model)
    
    model = Dense(128, activation = "relu")(model)
    model = Dropout(0.5)(model)
    
    model = Dense(256, activation = "relu")(model)
    model = Dropout(0.5)(model)
    
    model = Dense(3, activation = "softmax")(model)
    
    return model

def compile_model():
	in_layer = Input((224,224,4))
	pred_layer = model_build(in_layer, 4)
	model = Model(input=in_layer, output = pred_layer)
	model.compile(optimizer=RMSprop(), loss="categorical_crossentropy", metrics=["accuracy"])
	return model