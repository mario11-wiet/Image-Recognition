from keras.models import load_model

arr_category = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truct']
path = 'C:/Users/mk110/Desktop/lo'
model = load_model('save_model/myModel.h5')
# https://keras.io/api/datasets/cifar10/
# https://www.cs.toronto.edu/~kriz/cifar.html