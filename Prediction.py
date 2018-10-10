import cv2
import tensorflow as tf

CATEGORIES = ["Dog", "Cat"]  # will use this to convert prediction num to string value


def prepare(filepath):
    IMG_SIZE = 50  # 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # return the image with shaping that TF wants.


model = tf.keras.models.load_model("1*32*1-CNN.model")

test1 = prepare("images/dog1.jpg")
test2 = prepare("images/cat1.jpg")
test3 = prepare("images/cat.jpeg")
test4 = prepare("images/dog.jpeg")

testList = [test1,test2,test3,test4]

for i in testList:


    prediction = model.predict([i])  # REMEMBER YOU'RE PASSING A LIST OF THINGS YOU WISH TO PREDICT

    print(prediction)  # will be a list in a list.
    print(CATEGORIES[int(prediction[0][0])])
