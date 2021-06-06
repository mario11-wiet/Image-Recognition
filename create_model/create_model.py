import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras import backend as K
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.constraints import maxnorm
import model_constants as mc


class create_model:
    def __init__(self, model):
        # wczytanie zdjec do treningu i uczenia
        (self.X_train, self.y_train), (self.X_test, self.y_test) = cifar10.load_data()
        self.model = model

    def dateset(self):
        # konwertowanie (przerabianie tych zdjęć), aby wartości (normalnie 0-255 (kolor, na pikselu) były w przedziale 0-1)
        # konwertowanie inta na float (domyślnie 32)
        self.X_train = self.X_train.astype('float32')
        self.X_test = self.X_test.astype('float32')
        # to dzielenie o którym pisałem wyżej
        self.X_train = self.X_train / 255.0
        self.X_test = self.X_test / 255.0
        # konwertowanie wektora klasy, na binarną macierz
        # zamiast [0,1,2] mamy [[1,0,0],[0,1,0],[0,0,1]]
        self.y_train = np_utils.to_categorical(self.y_train)
        self.y_test = np_utils.to_categorical(self.y_test)

    def build_model(self):
        # inicjalizacja modelu
        self.model = Sequential()
        # dodawanie wart modelu
        # 32 - rozmiar filtra
        # (3, 3) - kształ jądra splotowego
        # same - brak zmiany rozmiaru
        self.model.add(Conv2D(32, (3, 3), input_shape=self.X_train.shape[1:], activation='relu', padding='same'))
        # odrzucienie 25% połączeń
        self.model.add(Dropout(0.15))
        # Maksymalna operacja łączenia danych przestrzennych 2D, próbkuje dane wejściowe wzdłuż jego wymiarów przestrzennych (wysokość i szerokość
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # normalizacja danych przed kolejną wartwą
        self.model.add(BatchNormalization())

        # to samo co wyżej, ale filtr 64
        self.model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        self.model.add(Dropout(0.15))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(BatchNormalization())

        # to samo co wyżej, ale filtr 128
        self.model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        self.model.add(Dropout(0.15))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(BatchNormalization())

        # kończąc z wartwami splotowymi potrzebujemy spłaszczyć dane
        self.model.add(Flatten())
        self.model.add(Dropout(0.15))

        # dense - odpowiada za neuroony w gęstych wartwach, pierwszy parametr- ilość neutronów, 2 odpowiada za brak nadmiernych dopasowań, 3 zdany
        self.model.add(Dense(256, kernel_constraint=maxnorm(3)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.20))
        self.model.add(BatchNormalization())
        # zmiejszanie liczby neurtonów w kolejnych wartwach

        self.model.add(Dense(128, kernel_constraint=maxnorm(3)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.20))
        self.model.add(BatchNormalization())

        self.model.add(Dense(64, kernel_constraint=maxnorm(3)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.20))
        self.model.add(BatchNormalization())

        self.model.add(Dense(32, kernel_constraint=maxnorm(3)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.20))
        self.model.add(BatchNormalization())

        self.model.add(Dense(mc.numer_class))
        # wybiera nautron o największym prawdopodobienstwu
        self.model.add(Activation('softmax'))

        # kompilacja moddelu
        self.model.compile(loss='categorical_crossentropy', optimizer=mc.optimizer, metrics=['accuracy'])

    def train_model(self):
        self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=mc.epoch,
                       batch_size=64)

    def accuraty_model(self):
        scores = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))
        #Skoteczność modelu
        #Accuracy: 77.49%

if __name__ == '__main__':
    myModel = None
    tf.keras.backend.clear_session()
    model = create_model(myModel)
    model.dateset()
    model.build_model()
    model.train_model()
    model.accuraty_model()
    myModel = model.model
    myModel.save(mc.ML_MODEL_FILENAME)
