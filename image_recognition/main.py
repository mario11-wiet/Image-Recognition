import numpy as np
from keras.preprocessing import image
import constants as c

def main():
    #wczytanie zdjecia w rozmiarze
    img = image.load_img(c.path, target_size=(32, 32))
    #przetwarzanie zdjecia
    processing = image.img_to_array(img)
    processing = np.expand_dims(processing, axis=0)
    category = c.model.predict(processing)



if __name__ == '__main__':
    main()
