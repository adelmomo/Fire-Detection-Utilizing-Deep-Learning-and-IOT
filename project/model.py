from tensorflow.keras.layers import *
from tensorflow.keras import Model
import numpy as np
import cv2
import os
from tensorflow.keras.callbacks import LearningRateScheduler 
from tensorflow.keras.utils import plot_model
import sklearn
import tensorflow
import matplotlib as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
path='fire\\'
n,m,c=100,100,3
##X=np.zeros((0,n,m,c),dtype='float32')
##Y=np.zeros((0,1),dtype='float32')
##for f in os.listdir(path):
##    img=cv2.imread(path+f)
##    img=cv2.resize(img,(n,m))
##    X=np.append(X,img.reshape(1,n,m,c),axis=0)
##    Y=np.append(Y,[[1]],axis=0)
##path='non-fire\\'
##for f in os.listdir(path):
##    
##    img=cv2.imread(path+f)
##    img=cv2.resize(img,(n,m))
##    X=np.append(X,img.reshape(1,n,m,c),axis=0)
##    Y=np.append(Y,[[0]],axis=0)
##Y=tensorflow.keras.utils.to_categorical(Y)

#trainx,testx,trainy,testy=train_test_split(X,Y,test_size=0.2,random_state=100)
mean=np.load('mean.npy')
std=np.load('Std.npy')

#np.save('mean.npy',mean)
#np.save('Std.npy',std)

input_layer=Input(shape=(n,m,c))
x=Conv2D(16,(3,3))(input_layer)
x=BatchNormalization()(x)
x=tensorflow.keras.layers.ReLU()(x)
x=MaxPooling2D((3,3))(x)
x=Dropout(0.2)(x)
x=Conv2D(32,(3,3))(x)
x=BatchNormalization()(x)
x=tensorflow.keras.layers.ReLU()(x)
x=MaxPooling2D((3,3))(x)
x=Dropout(0.2)(x)
x=Conv2D(64,(3,3))(x)
x=BatchNormalization()(x)
x=tensorflow.keras.layers.ReLU()(x)
x=Dropout(0.2)(x)
x=Conv2D(128,(3,3))(x)
x=BatchNormalization()(x)
x=tensorflow.keras.layers.ReLU()(x)
x=MaxPooling2D((3,3))(x)

##x=Conv2D(512,(3,3))(x)
##x=BatchNormalization()(x)
##x=Activation('relu')(x)
x=Dropout(0.2)(x)
x=Flatten()(x)
x=Dense(1024)(x)
x=BatchNormalization()(x)
x=tensorflow.keras.layers.ReLU()(x)
x=Dropout(0.2)(x)
x=Dense(2)(x)
x=BatchNormalization()(x)
x=Activation('softmax')(x)
model=Model(inputs=input_layer,outputs=x)
callbacks=tensorflow.keras.callbacks.ModelCheckpoint(filepath='fire-fold.h5', verbose=1, save_best_only=True,monitor='val_acc')
batch=32
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
#history=model.fit(trainx,trainy,callbacks=[callbacks],validation_data=(testx,testy),epochs=100,verbose=2)
model.load_weights('fire-detect11.h5')
#history=model.fit_generator(datagen.flow(trainx,trainy,batch_size=batch),steps_per_epoch=len(trainx)//batch,validation_data=testgen.flow(testx,testy,batch_size=batch),validation_steps=len(testx)//batch,callbacks=callbacks,epochs=60,verbose=2)
##import numpy as np
##import matplotlib.pyplot as plt
##
##from sklearn import svm, datasets
##from sklearn.model_selection import train_test_split
##from sklearn.metrics import confusion_matrix
##from sklearn.utils.multiclass import unique_labels
##def plot_confusion_matrix(y_true, y_pred, classes,
##                          normalize=False,
##                          title=None,
##                          cmap=plt.cm.Blues):
##    """
##    This function prints and plots the confusion matrix.
##    Normalization can be applied by setting `normalize=True`.
##    """
##    if not title:
##        if normalize:
##            title = 'Normalized confusion matrix'
##        else:
##            title = 'Confusion matrix, without normalization'
##
##    # Compute confusion matrix
##    cm = confusion_matrix(y_true, y_pred)
##    # Only use the labels that appear in the data
##    
##    if normalize:
##        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
##        print("Normalized confusion matrix")
##    else:
##        print('Confusion matrix, without normalization')
##
##    print(cm)
##
##    fig, ax = plt.subplots()
##    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
##    ax.figure.colorbar(im, ax=ax)
##    # We want to show all ticks...
##    ax.set(xticks=np.arange(cm.shape[1]),
##           yticks=np.arange(cm.shape[0]),
##           # ... and label them with the respective list entries
##           xticklabels=classes, yticklabels=classes,
##           title=title,
##           ylabel='True label',
##           xlabel='Predicted label')
##
##    # Rotate the tick labels and set their alignment.
##    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
##             rotation_mode="anchor")
##
##    # Loop over data dimensions and create text annotations.
##    fmt = '.2f' if normalize else 'd'
##    thresh = cm.max() / 2.
##    for i in range(cm.shape[0]):
##        for j in range(cm.shape[1]):
##            ax.text(j, i, format(cm[i, j], fmt),
##                    ha="center", va="center",
##                    color="white" if cm[i, j] > thresh else "black")
##    fig.tight_layout()
##    return ax
##
##class_names=['Fire','Non-Fire']
##np.set_printoptions(precision=2)
##y=model.predict(testx)
##y=y.argmax(axis=1)
### Plot non-normalized confusion matrix
##plot_confusion_matrix(testy.argmax(axis=1),y,classes=class_names,
##                      title='Confusion matrix')
##
### Plot normalized confusion matrix
##
##
##plt.show()
