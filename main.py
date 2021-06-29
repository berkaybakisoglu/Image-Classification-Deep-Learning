import matplotlib.pyplot as plt
import os
import random
import sys
import sns as sns
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix
import sklearn
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from numpy.random import seed
import seaborn as sns
from PyQt5.QtWidgets import (QPushButton, QWidget,QMainWindow,
    QLineEdit, QApplication)
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon

seed(1)
tf.random.set_seed(1)
imagesDir = r"C:\Users\user\Desktop\Dataset\RaporTrain"
testDir = r"C:\Users\user\Desktop\Dataset\RaporTest"
validDir= r"C:\Users\user\Desktop\Dataset\RaporValid"
cropImagesDir =  r"C:\Users\user\Desktop\Dataset\RaporCropped"
artistDictionary = {}
batchSize = 64
trainShape = (224,224, 3)
def get_random_crop(image, crop_height, crop_width):

    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    crop = image[y: y + crop_height, x: x + crop_width]

    return crop
def getClassWeights():
    global trainClasses
    artistDirectories = os.listdir(cropImagesDir)
    for artist in artistDirectories:
        newDirectory=os.listdir(os.path.join(cropImagesDir,artist))
        for file in newDirectory:
            # x = random_image_file[(len(newImagesDir) + 1):].split("_")
            if artist not in artistDictionary.keys():
                artistDictionary[artist]=1
            else:
                artistDictionary[artist]+=1

    print(sorted(artistDictionary.items(), key=lambda x: x[1],reverse=True))
    totalPaintings=sum(artistDictionary.values())
    keys=artistDictionary.keys()
    totalArtists= len(keys)
    classWeights=dict()
    print(totalPaintings)
    for key in keys:
        if key not in classWeights.keys():
            classWeights[key]=totalPaintings/(totalArtists*artistDictionary[key])

    trainClasses=len(artistDictionary.keys())
    print(sorted(classWeights.items(), key=lambda x: x[1], reverse=True))
    print(sum(classWeights.values()))
    return classWeights
def checkDictionariesValid():
    for name in list(artistDictionary.keys()):
        if os.path.exists(os.path.join(imagesDir, name)):
            print("Found -->", os.path.join(imagesDir, name))
        else:
            print("Did not find -->", os.path.join(imagesDir, name))
def getRandomImages():
    fig, axes = plt.subplots(1, 5, figsize=(20, 10))
    for i in range(5):
        print(type(artistDictionary.keys()))
        random_artist = random.choice(list(artistDictionary.keys()))
        random_image = random.choice(os.listdir(os.path.join(imagesDir, random_artist)))
        random_image_file = os.path.join(imagesDir, random_artist, random_image)
        image = plt.imread(random_image_file)
        axes[i].imshow(image)
        axes[i].set_title("Artist: " + random_artist.replace('_', ' '))
        axes[i].axis('off')
    plt.show()


def readImages():
    # checkDictionariesValid()
    # getRandomImages()
    trainDatagen = ImageDataGenerator(validation_split=0,
                                      horizontal_flip=True,
                                      vertical_flip=True,
                                      zoom_range=0.2,
                                      rotation_range=45,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      shear_range=0.2,
                                      preprocessing_function=preprocess_input
                                      )
    validationDataGen = ImageDataGenerator(validation_split=0,preprocessing_function=preprocess_input)
    trainDataset = trainDatagen.flow_from_directory(directory=cropImagesDir,
                                                    # directory=imagesDir,
                                                    class_mode='categorical',
                                                    target_size=trainShape[0:2],
                                                    color_mode="rgb",
                                                    batch_size=batchSize,
                                                    subset="training",
                                                    shuffle=True,
                                                    classes=list(artistDictionary.keys()),
                                                    )
    validationDataset =  validationDataGen.flow_from_directory(directory=testDir,
                                                         class_mode='categorical',
                                                         target_size=trainShape[0:2],
                                                         color_mode="rgb",
                                                         batch_size=batchSize,
                                                         shuffle=True,
                                                         classes=list(artistDictionary.keys()),

                                                         )

    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=trainShape)
    for layer in base_model.layers:
        layer.trainable = False
    model = tf.keras.models.Sequential()
    model.add(base_model)
    model.add(tf.keras.layers.Flatten())
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(trainClasses, activation='softmax'))
    # model=tf.keras.models.Sequential()
    # model.add(tf.keras.layers.Conv2D(56, (3, 3), activation='relu', input_shape=(224,224, 3)))
    # model.add(tf.keras.layers.MaxPool2D(2,2))
    # model.add(tf.keras.layers.Conv2D(112, (3, 3), activation='relu'))
    # model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
    # model.add(tf.keras.layers.Conv2D(224, (3, 3), activation='relu'))
    # model.add(tf.keras.layers.Flatten())
    # model.add(tf.keras.layers.Dense(trainClasses, activation='softmax'))
    model.summary()
    optimizer = Adam(lr=0.0009)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(3)])
    n_epoch = 40
    early_stop = EarlyStopping(monitor='val_loss', patience=6, verbose=1,
                               mode='auto', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=6,
                                  verbose=1, mode='auto')
    STEP_SIZE_TRAIN = trainDataset.n // trainDataset.batch_size+1
    STEP_SIZE_VALID = validationDataset.n // validationDataset.batch_size+1
    print("Total number of batches =", STEP_SIZE_TRAIN, "and", STEP_SIZE_VALID)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='{epoch:08d}.h5', period=5)
    history1 = model.fit_generator(generator=trainDataset, steps_per_epoch=STEP_SIZE_TRAIN,
                                   validation_data=validationDataset, validation_steps=STEP_SIZE_VALID,
                                   epochs=n_epoch,
                                   shuffle=True,
                                   verbose=1,
                                   callbacks=[reduce_lr, checkpoint, early_stop],
                                   )
    plt.plot(history1.history['accuracy'])
    plt.plot(history1.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history1.history['loss'])
    plt.plot(history1.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history1.history['top_k_categorical_accuracy'])
    plt.plot(history1.history['val_top_k_categorical_accuracy'])
    plt.title('model TOP3')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    model.save("TL_CROPPED_MODEL")

    trainDataset = trainDatagen.flow_from_directory(directory=imagesDir,
                                                    class_mode='categorical',
                                                    target_size=trainShape[0:2],
                                                    color_mode="rgb",
                                                    batch_size=batchSize,
                                                    subset="training",
                                                    shuffle=True,
                                                    classes=list(artistDictionary.keys())
                                                    )
    STEP_SIZE_TRAIN = trainDataset.n // trainDataset.batch_size
    history2 = model.fit_generator(generator=trainDataset, steps_per_epoch=STEP_SIZE_TRAIN,
                                   validation_data=validationDataset, validation_steps=STEP_SIZE_VALID,
                                   epochs=n_epoch,
                                   shuffle=True,
                                   verbose=1,
                                   callbacks=[reduce_lr, checkpoint, early_stop],
                                   )
    plt.plot(history2.history['accuracy'])
    plt.plot(history2.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history2.history['loss'])
    plt.plot(history2.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history2.history['top_k_categorical_accuracy'])
    plt.plot(history2.history['val_top_k_categorical_accuracy'])
    plt.title('model TOP3')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    model.save("TLSONDENEME+CROPPED_MODEL")
    #test_generator = ImageDataGenerator().flow_from_directory(
    #    testDir,
    #    target_size=trainShape[0:2],
    #    batch_size=batchSize,
    #    class_mode='categorical')
    #predictions = model.predict_generator(test_generator, validationDataset.n // validationDataset.batch_size)
    #print('predictions shape:', predictions.shape)
    #print('predictions:', predictions)

    # getting the labels



getClassWeights()
#readImages()
def showClassficationReport_Generator(model, valid_generator, STEP_SIZE_VALID):
    chck=0
    tick_labels = artistDictionary.keys()
    y_pred, y_true,y_pred3 = [], [],[]
    for i in range(STEP_SIZE_VALID):
        (X, y) = next(valid_generator)
       # if chck<10:
        #    chck+=1
        #    image = X[i]
        #    plt.imshow(image)
        #    plt.show()

        y_pred.append(model.predict(X))
        y_true.append(y)

    # Create a flat list for y_true and y_pred
    y_pred = [subresult for result in y_pred for subresult in result]
    y_true = [subresult for result in y_true for subresult in result]

    # Update Truth vector based on argmax
    y_true = np.argmax(y_true, axis=1)
    y_true = np.asarray(y_true).ravel()

    # Update Prediction vector based on argmax
    y_pred = np.argmax(y_pred, axis=1)
    y_pred = np.asarray(y_pred).ravel()

    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    conf_matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(artistDictionary)))
    conf_matrix = conf_matrix / np.sum(conf_matrix, axis=1)
    sns.heatmap(conf_matrix, annot=True, fmt=".2f", square=True, cbar=False,
                cmap=plt.cm.jet, xticklabels=tick_labels, yticklabels=tick_labels, ax=ax)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title('Confusion Matrix')
    plt.show()

    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=np.arange(len(artistDictionary)), target_names=artistDictionary.keys()))




#valDatagen = ImageDataGenerator(validation_split=0,preprocessing_function=preprocess_input)


#loadedModel = tf.keras.models.load_model(r'C:\Users\user\PycharmProjects\pythonProject2\TLSONDENEME+CROPPED_MODEL')
#loadedModel.summary()
#validationDataset = valDatagen.flow_from_directory(directory=testDir,class_mode='categorical',target_size=trainShape[0:2],color_mode="rgb",batch_size=batchSize,shuffle=False,classes=list(artistDictionary.keys()),)
#predictions = loadedModel.predict(validationDataset)
#STEP_SIZE_VALID = validationDataset.n // validationDataset.batch_size+1
#showClassficationReport_Generator(loadedModel,validationDataset, STEP_SIZE_VALID)
class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()

        self.setAlignment(Qt.AlignLeft)
        self.setText('\n\n Drop Image Here \n\n')
        self.setStyleSheet('''
            QLabel{
                border: 12px dashed #aaa
            }
        ''')

    def setPixmap(self, image):
        super().setPixmap(image)

class AppDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(1600, 1200)
        self.setAcceptDrops(True)
        self.loadedModel = tf.keras.models.load_model( r'C:\Users\user\PycharmProjects\pythonProject2\DUZMODELDUZ+CROPPED_MODEL')
        self.tlModel = tf.keras.models.load_model(r'C:\Users\user\PycharmProjects\pythonProject2\TLSONDENEME+CROPPED_MODEL')
        self.mainLayout = QHBoxLayout()
        self.l0 = QLabel()
        self.l1 = QLabel()
        self.l2 = QLabel()
        self.l3 = QLabel()
        self.l4 = QLabel()
        self.l5 = QLabel()
        self.l6 = QLabel()
        self.l7 = QLabel()
        self.l0.setText("Baseline Resnet50")
        self.l1.setText("Best Prediction")
        self.l2.setText("Second Prediction")
        self.l3.setText("Third Prediction")
        self.l4.setText("Transfer Learning Resnet50")
        self.l5.setText("Best Prediction")
        self.l6.setText("Second Prediction")
        self.l7.setText("Third Prediction")
        myFont = QFont()
        myFont.setBold(True)
        self.l0.setFont(myFont)
        self.l4.setFont(myFont)

        self.layout1=QVBoxLayout()
        self.layout2=QVBoxLayout()
        self.photoViewer = ImageLabel()
        self.photoViewer.setGeometry(0,10,500,500)
        self.mainLayout.addWidget(self.photoViewer)
        self.layout1.addWidget(self.l0)
        self.layout1.addWidget(self.l1)
        self.layout1.addWidget(self.l2)
        self.layout1.addWidget(self.l3)
        self.layout2.addWidget(self.l4)
        self.layout2.addWidget(self.l5)
        self.layout2.addWidget(self.l6)
        self.layout2.addWidget(self.l7)

        self.mainLayout.addLayout(self.layout1)
        self.mainLayout.addLayout(self.layout2)
        self.setLayout(self.mainLayout)

    def dragEnterEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):

        if event.mimeData().hasImage:
            y_pred = []
            classes = np.array(list(artistDictionary.keys()))
            event.setDropAction(Qt.CopyAction)
            file_path = event.mimeData().urls()[0].toLocalFile()
            self.set_image(file_path)

            test_image = image.load_img(file_path,target_size=(224,224))
            test_image = image.img_to_array(test_image)
            #test_image = get_random_crop(test_image, 224, 224)
            test_image2=test_image
            test_image = test_image / 255
            test_image = np.expand_dims(test_image, axis=0)
            test_image2 = np.expand_dims(test_image2, axis=0)
            y_pred=self.loadedModel.predict(test_image)
            y_pred = [subresult for result in y_pred for subresult in result]
            y_pred=np.array(y_pred)
            total=np.array(y_pred).sum()
            bestpred = np.argmax(y_pred, axis=0)
            print(bestpred)
            print(y_pred[bestpred]/total,classes[bestpred])
            self.l1.setText(classes[bestpred] + " " + str("{:.2f}".format(y_pred[bestpred]/total )))
            y_pred=np.delete(y_pred,bestpred)
            classes=np.delete(classes,bestpred)
            bestpred = np.argmax(y_pred, axis=0)
            print(y_pred[bestpred] / total, classes[bestpred])
            self.l2.setText(classes[bestpred] + " " + str("{:.2f}".format(y_pred[bestpred]/total )))
            y_pred=np.delete(y_pred, bestpred)
            classes=np.delete(classes, bestpred)
            bestpred = np.argmax(y_pred, axis=0)
            print(y_pred[bestpred] / total, classes[bestpred])
            self.l3.setText(classes[bestpred] + " " + str("{:.2f}".format(y_pred[bestpred]/total )))

            y_pred = []
            classes = np.array(list(artistDictionary.keys()))
            y_pred = self.tlModel.predict(test_image2)
            y_pred = [subresult for result in y_pred for subresult in result]
            y_pred = np.array(y_pred)
            total = np.array(y_pred).sum()
            bestpred = np.argmax(y_pred, axis=0)
            print(bestpred)
            print(y_pred[bestpred] / total, classes[bestpred])
            self.l5.setText(classes[bestpred] + " " + str("{:.2f}".format(y_pred[bestpred]/total )))
            y_pred = np.delete(y_pred, bestpred)
            classes = np.delete(classes, bestpred)
            bestpred = np.argmax(y_pred, axis=0)
            print(y_pred[bestpred] / total, classes[bestpred])
            self.l6.setText(classes[bestpred] + " " + str("{:.2f}".format(y_pred[bestpred]/total )))
            y_pred = np.delete(y_pred, bestpred)
            classes = np.delete(classes, bestpred)
            bestpred = np.argmax(y_pred, axis=0)
            print(y_pred[bestpred] / total, classes[bestpred])
            self.l7.setText(classes[bestpred] + " " + str("{:.2f}".format(y_pred[bestpred]/total )))
            #print(total,y_pred,artistDictionary.keys())
            #y_pred = np.asarray(y_pred).ravel()
            #print(y_pred[bestpred]/percent,list(artistDictionary)[bestpred[0]])

            event.accept()
        else:
            event.ignore()

    def set_image(self, file_path):
        pixmap=QPixmap(file_path)
        pixmap = pixmap.scaledToWidth(512)
        self.photoViewer.setPixmap(pixmap)




app = QApplication(sys.argv)
demo = AppDemo()
demo.show()
sys.exit(app.exec_())