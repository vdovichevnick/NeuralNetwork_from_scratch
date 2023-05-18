import numpy as np
from PIL import Image
import climage
import pickle


def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    """
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache


def relu(Z):
    """
    Implement the RELU function.
    """
    A = np.maximum(0, Z)
    cache = Z
    return A, cache


def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.
    """
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer
    """

    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    """

    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")
    caches.append(cache)

    return AL, caches


def predict(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.
    """

    m = X.shape[1]
    n = len(parameters) // 2  # number of layers in the neural network
    p = np.zeros((1, m))

    probas, caches = L_model_forward(X, parameters) # Forward propagation

    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    print("Accuracy: " + str(np.sum((p == y) / m)))

    return p


# loading a dictionary from a file
with open("weights.pkl", "rb") as file:
    parameters = pickle.load(file)

classes = [b'non-cat', b'cat']
print(classes)
num_px = 64
n_x = 12288  # num_px * num_px * 3
n_h = 7
n_y = 1
layers_dims = [n_x, n_h, n_y]  # 2-layer model

## Check with your 1st image
my_label_y = [1]  # the true class of your image (1 -> cat, 0 -> non-cat)
fname = "/Users/nikitavdovichev/Downloads/3.jpg"  # change this to the name of your image path
image = np.array(Image.open(fname).resize((num_px, num_px)))
output = climage.convert(fname, is_unicode=True)
print(output)
image = image / 255.
image = image.reshape((1, num_px * num_px * 3)).T
print("In your attempt:")
my_predicted_image = predict(image, my_label_y, parameters)

print("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[
    int(np.squeeze(my_predicted_image))].decode("utf-8") + "\" picture.")

## Check with your 2nd image
my_label_y = [0]  # the true class of your image (1 -> cat, 0 -> non-cat)
fname = "/Users/nikitavdovichev/Downloads/car.jpg"  # change this to the name of your image path
image = np.array(Image.open(fname).resize((num_px, num_px)))
output = climage.convert(fname, is_unicode=True)
print(output)
image = image / 255.
image = image.reshape((1, num_px * num_px * 3)).T
print("In your attempt:")
my_predicted_image = predict(image, my_label_y, parameters)

print("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[
    int(np.squeeze(my_predicted_image))].decode("utf-8") + "\" picture.")
## Check with your 3rd image
my_label_y = [0]  # the true class of your image (1 -> cat, 0 -> non-cat)
fname = "/Users/nikitavdovichev/Downloads/dog.jpeg"  # change this to the name of your image path
image = np.array(Image.open(fname).resize((num_px, num_px)))
output = climage.convert(fname, is_unicode=True)
print(output)
image = image / 255.
image = image.reshape((1, num_px * num_px * 3)).T
print("In your attempt:")
my_predicted_image = predict(image, my_label_y, parameters)

print("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[
    int(np.squeeze(my_predicted_image))].decode("utf-8") + "\" picture.")
