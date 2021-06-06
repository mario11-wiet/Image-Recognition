from pprint import pprint
import numpy as np
import constants as c
from keras.models import load_model
from keras.preprocessing import image
model = load_model('save_model/myModel.h5')


def main():
    # wczytanie zdjecia w rozmiarze
    img = image.load_img(c.path, target_size=(32, 32))
    img_l = img.convert("L")
    img_s = img.convert("1")
    img.show()
    img_s.show()
    img_l.show()
    # przetwarzanie zdjecia
    processing = image.img_to_array(img)
    processing = np.expand_dims(processing, axis=0)
    category = model.predict(processing)
    category = list(enumerate(category[0]))
    category = sorted(category, key=lambda x: x[1], reverse=True)
    pprint(c.arr_category[category[0][0]])


if __name__ == '__main__':
    main()
