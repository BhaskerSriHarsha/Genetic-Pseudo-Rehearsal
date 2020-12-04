import requests
import time
import numpy as np
import tensorflow as tf


# GLOBAL VARIABLES
# Also called as "status_flag" in the companion paper.
training_flag = "0"


# STEP 1: LOADING THE MODEL FROM DISK **************************
print("Loading the model..")
loaded_model = tf.keras.models.load_model("ML_Model")
# loaded_model = tf.keras.models.load_model("ML_MODEL_2")
print("Loaded the model from disk \n\n")
print(loaded_model.summary())

# STEP 2: SEND REQUEST TO BEGIN TRAINING ***************************
print("Sending request to begin training")
test_data = "0,1"
try:
    requests.post("http://127.0.0.1:5000/",data=test_data, timeout=1)
except:
    print("Request completed \n")

# STEP 3: FETCHING TRAINING FLAG *************************************
fetched_training_flag = requests.get("http://127.0.0.1:5000/training")
print("Fetched the trianing flag: ", fetched_training_flag.text, type(fetched_training_flag.text))
training_flag = fetched_training_flag.text
print("Training flag updated to: ",training_flag)

while training_flag=="1":# or also status_flag

    print("\n \n Requesting flag variable")
    flag = requests.get("http://127.0.0.1:5000/flag")
    # this flag was named as "images_flag" in the companion paper and the original algorithm. Will be changed in the future.
    print("received flag variable ", flag.text)

    print("\nChecking if flag variable is true")
    if flag.text=="1":
        print("Flag variable is true")

        # get the images from server and print them
        images = requests.get("http://127.0.0.1:5000/images")
        images = np.fromstring(images.content, np.float64)
        images = np.reshape(images,(8,28,28))

        print("Shape of received images: ", images.shape)

        predictions = loaded_model.predict(images)
        predictions = predictions.tostring()
        r = requests.post("http://127.0.0.1:5000/predictions",data=predictions)
        print("Predictions sent to server")

        # set flag = 0 set ready = 1
        re_fl = requests.post("http://127.0.0.1:5000/reset_flag",data="0")
        red_fl = requests.post("http://127.0.0.1:5000/ready",data="1")

    time.sleep(1)

    # CHECKING THE TRAINING FLAG
    '''training_flag is being constantly checked as it is the loop variable.'''
    fetched_training_flag = requests.get("http://127.0.0.1:5000/training")
    training_flag = fetched_training_flag.text

# REQUESTING THE SYNTHETIC DATA FROM THE server
synthetic_data = requests.get("http://127.0.0.1:5000/synthetic_data")
synthetic_data = np.fromstring(synthetic_data.content, np.float64)
synthetic_data = np.reshape(synthetic_data,(16,28,28))
print("\n Synthetic data received from server")
print("Shape:: ", synthetic_data.shape)
