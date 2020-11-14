from flask import Flask, request
import time
import numpy as np
from scipy.spatial import distance
from sklearn.mixture import GaussianMixture
from heapq import nlargest
import random
from Genetic_Rehearsal import *
import copy


'''
VERSION 3.0: Implementation of Genetic rehearsal algorithm in entirity.
'''


app = Flask(__name__)

# Declare the global variables here
'''
flag_variable is used by client to see if images are ready for taking.
ready_flag is used by "server" to know whether predictions are ready on the client side to be collected.
training is used to stop the training process loop on the client side.
'''
flag_variable="0"
ready_flag="0"
training_flag="0"
predictions = "nothing"
synthetic_data = 0

def AddTuple(TUPLE, ELEMENT):
    TEMP = list(TUPLE)
    TEMP.insert(0, ELEMENT)

    return tuple(TEMP)


# Declare the main training function here
@app.route('/', methods=['POST'])
def Start():
    raw_data = str(request.data)
    cleaned_data = raw_data[2:len(raw_data)-1]
    split_data = cleaned_data.split(",")
    received_data = [int(x) for x in split_data]
    print("Final received labels: ", received_data)


    global flag_variable, images_variable, ready_flag, training_flag, synthetic_data, predictions

    # Declaring the varibles required before hand. In the later versions, these will arrive as arguments.
    SHAPE = (28,28)
    TARGET_CLASSES = received_data
    verbose=1
    population_size=8
    NUMBER_OF_CULTURES = 1
    NUMBER_OF_GENERATIONS = 100
    MUTATION_PROBABILITY = 0.6
    MUTATION_TYPE = "+"

    print("Set the training_flag to 1")
    training_flag = "1"


    # Declaring the variable
    temp_shape = AddTuple(SHAPE,0)
    pseudo_x_train_2400 = np.zeros(temp_shape)

    POINTS_GATHERED = []
    ideal_points = []

    for class_number in TARGET_CLASSES:
        POINTS_COUNTER=0
        if verbose == 1:
            print("\n\n\n ****************************************************************")
            print("Class number: ", class_number)
            print("****************************************************************")

        for p in range(NUMBER_OF_CULTURES):
            '''CREATING THE INITIAL POPULATION'''
            current_generation=[]
            for i in range(population_size):
                current_generation.append(np.zeros(SHAPE))
            '''Declaring the stopping criteria for the genetic algorithm.'''
            best = 0
            best_list = []

            while best*100 < 99:
            # for generation_number in range(3):
                current_generation_fitness=[]

                # 1. CLUB THE ENTIRE GENERATION INTO ONE NUMPY ARRAY
                ''' First we need to convert all the numpy arrays in current_generation to (1,28,28) shape.'''
                temp_shape = AddTuple(SHAPE,1)
                for i in range(population_size):
                    current_generation[i] = np.reshape(current_generation[i],temp_shape)

                images_variable = np.concatenate(current_generation,axis=0)
                # print("SHape of the images data : ", images_variable.shape)

                # 2. CONVERT IT TO string
                images_variable = images_variable.tostring()

                # 3. SEND THE NUMPY ARRAY TO CLIENT
                flag_variable = "1"

                # 4. RECEIVE THE PREDICTIONS FROM CLIENT
                while ready_flag=="0":
                    time.sleep(1)
                # print("Shape of received predictions is: ", predictions.shape)
                predictions = np.reshape(predictions,(8,10))
                # print("Reshaped predictions: ",predictions.shape)

                # 5. UNWRAP THE PREDICTIONS AND APPEND TO current_generation_fitness list
                for i in range(population_size):
                    class_prediction= np.argmax(predictions[i], axis=-1)
                    fitness_score = predictions[i][class_number]

                    current_generation_fitness.append(fitness_score)

                ready_flag="0"
                flag_variable = "0"

                if max(current_generation_fitness) >= best:
                    best = max(current_generation_fitness)
                if verbose == 1:
                    print("Best: ",best*100, " Culture number: ",p," Class number: ",class_number)

                fittest_four_model_indices = nlargest(int(population_size/4), range(len(current_generation_fitness)), current_generation_fitness.__getitem__)
                temp_list=[]
                for temp in range(int(population_size/4)):
                    temp_list.append(np.copy(current_generation[fittest_four_model_indices[temp]]))

                for temp in range(int(population_size/4)):
                    current_generation[temp] = np.copy(temp_list[temp])

                for temp in range((int(population_size/4)),(int(population_size/2))):
                    current_generation[temp] = np.copy(current_generation[temp-(int(population_size/4))])

                for i in range((int(population_size/4)),(int(population_size/2))):
                    point_mutation=np.random.choice([0,abs(np.random.normal(0,1,[1]))],size=current_generation[i].shape, p=[(1-MUTATION_PROBABILITY),MUTATION_PROBABILITY])
                    point_mutation = np.float64(point_mutation)
                    if MUTATION_TYPE == "*":
                        choice=random.choice(["+","-","*"])
                        if choice== "-":
                            current_generation[i] = current_generation[i] - point_mutation
                        elif choice=="+":
                            current_generation[i] = current_generation[i] + point_mutation
                    elif MUTATION_TYPE == "+":
                        current_generation[i] = current_generation[i] + point_mutation
                    else:
                        current_generation[i] = current_generation[i] - point_mutation

                for temp in range((int(population_size/2)),(int(population_size*0.75))):
                    current_generation[temp] = np.copy(current_generation[temp-(int(population_size/2))])

                for temp in range((int(population_size/2)),(int(population_size*0.75))):
                    if temp%2 == 0:
                        crossover(current_generation[temp],current_generation[temp+1],int((current_generation[temp].shape)[0]/2))

                for temp in range((int(population_size*0.75)),population_size):
                    current_generation[temp] = np.copy(current_generation[temp-(int(population_size/2))])
                for temp in range((int(population_size*0.75)),population_size):
                    if temp%2 == 0:
                        crossover(current_generation[temp],current_generation[temp+1],int((current_generation[temp].shape)[0]/2))

            for k in range(len(current_generation)):
                temp_shape = AddTuple(SHAPE,1)
                pseudo_x_train_2400 = np.vstack((pseudo_x_train_2400,np.reshape(current_generation[k],temp_shape)))

    print("Generated data: ",pseudo_x_train_2400.shape)

    if verbose == 1:
        print("Training data shape: ",pseudo_x_train_2400.shape)

    # Load the global variable with synthetic data
    print(pseudo_x_train_2400.dtype)
    synthetic_data = pseudo_x_train_2400.tostring()
    print("Converted the synthetic data to string.")
    # set the training flag to 0
    training_flag = "0"
    print("Set the training flag to : ", training_flag)

    return 1

# Declare the function that handles the Flag variable
@app.route('/flag',methods=['GET'])
def flag():
    # print("Client requested flag variable:: ", flag_variable)
    return flag_variable

# Function to reset the flag
@app.route('/reset_flag',methods=['POST'])
def reset_flag():
    global flag_variable
    flag_variable = request.data
    return "1"

# Declare the function that handles the ready variable
@app.route('/ready',methods=['POST'])
def ready():
    global ready_flag
    ready_flag = request.data
    return "1"


# Declare the function that sends images to the client
@app.route('/images',methods=['GET'])
def images():
    return images_variable

# declare the function that receives predictions from the client
@app.route('/predictions',methods=['POST'])
def fetch_predictions():
    global predictions
    # print("Predictions changed from", predictions)
    predictions = request.data
    predictions = np.fromstring(predictions, np.float32)
    # print("Predictions changed to", predictions)
    return "1"

@app.route('/training',methods=['GET'])
def training_status():
    global training_flag
    print("sent the training flag as : ", training_flag)
    return training_flag

@app.route('/synthetic_data', methods=['GET'])
def send_synthetic_data():
    return synthetic_data

if __name__ == "__main__":
    app.run(debug=True)
