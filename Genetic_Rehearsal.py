from scipy.spatial import distance
import numpy as np
import keras
from sklearn.mixture import GaussianMixture

def crossover(first_gene,second_gene,index):
  temp = np.copy(first_gene[index:])
  first_gene[index:] = second_gene[index:]
  second_gene[index:]=temp

def duplicate_remover(dataset,fitness_list):
  duplicate_count=0
  duplicate_index = [i for i in range(len(dataset))]

  for i in range(len(dataset)):
    if duplicate_index[i] != "D":
      j=i
      while j+1 < len(dataset):
        # if distance.euclidean(np.reshape(dataset[i],(784,)),np.reshape(dataset[j+1],(784,))) == 0:
        if distance.euclidean(np.reshape(dataset[i],(2,)),np.reshape(dataset[j+1],(2,))) == 0:
          duplicate_count += 1
          # print("Duplicate found @: ",j+1)
          fitness_list[j+1] = 0
          duplicate_index[j+1] = "D"
        j += 1

  return duplicate_count

def duplicate_counter(dataset):
  duplicate_count=0
  for i in range(len(dataset)):
    j=i
    while j+1 < len(dataset):
      if distance.euclidean(np.reshape(dataset[i],(784,)),np.reshape(dataset[j+1],(784,))) == 0:
        duplicate_count += 1
      j += 1

  return duplicate_count

def generate_labels(data,model,number_of_classes):
  '''
  . \n\n
  AUTHOR:
  Suri Bhasker Sri Harsha

  AIM:
  The function assigns labels to the given dataset with the given model as the reference.
  The function returns a list with labels in two formats. The first format is the one-hot encoding
  format and the second format is the binary format of labels.

  ARGUMENTS:
  data: The dataset for which you want to generate labels
  model: The model that will be used for the label generation
  number_of_classes: Number of classes in the dataset

  RETURNS:
  [labels, pre_labels]
  '''

  labels = model.predict_classes(data,verbose=1)
  pre_labels = labels
  labels = keras.utils.to_categorical(labels,number_of_classes)

  return [labels,pre_labels]

def confidence_filter(data, model, number_of_samples, verbose=0):

  '''
  . \n\n
  AUTHOR:
  Suri Bhasker Sri Harsha

  AIM:
  The function was developed to "filter" out points that are closer
  to the decision boudary compared to others. The function has the following arguments

  ARGUMENTS:
  data: The dataset in which you want to select the points closest to the decision boundary
  model: The model that will be used as the fitness function
  number_of_samples: Number of boundary points you want to achieve
  verbose: Default is 0. Displays the progress of the function
  '''
  predictions = model.predict(data,verbose=verbose)
  standard_deviations = np.std(predictions,axis=1)
  indices = np.argsort(standard_deviations)[:number_of_samples]

  print(indices.shape)
  return indices

def agreement_score(model1_predictions, model2_predictions):

  '''
  Returns the degree of agreement between predictions of two models

  INPUT ARGUMENTS
  model1_predictions: Preditions of the model 1 on a test dataset
  model2_predictions: Preditions of the model 2 on a test dataset

  OUTPUT:
  Returns a list of two elements where the first elements is the
  "Agreement score" between the two lists and the second element
  is the list of indices where both the models have agreed upon.
  '''

  if len(model1_predictions) != len(model2_predictions):
    print("Length of given lists donot match")
    return 0

  correct_count = 0
  agreement_indices = []

  for i in range(len(model1_predictions)):
    if model1_predictions[i] == model2_predictions[i]:
      correct_count += 1
      agreement_indices.append(i)

  agreement_score = (correct_count/len(model1_predictions))*100

  return [agreement_score, agreement_indices]


def Enrichment(data,labels,model,NUMBER_OF_CENTERS,NUMBER_OF_CLASSES, NUMBER_OF_SAMPLES):
    '''
    Enriches the given data by fitting a Gaussian Mixture model with
    NUMBER_OF_CENTERS and NUMBER_OF_SAMPLES
    '''
    gaussian = GaussianMixture(n_components = NUMBER_OF_CENTERS)
    gaussian.fit(X=data)
    synthetic = gaussian.sample(n_samples=NUMBER_OF_SAMPLES)

    synthetic_data = synthetic[0]

    labels = model.predict_classes(synthetic_data,verbose=1)
    pre_labels = labels
    labels = keras.utils.to_categorical(labels,NUMBER_OF_CLASSES)

    return [synthetic_data, labels, pre_labels]
def interleaved_rehearsal(model, NEW_DATA, REHEARSAL_DATA, OLD_TEST_DATA, NEW_TEST_DATA,epochs=20):

  '''

  Performs interleaved rehearsal of NEW_DATA and REHEARSAL_DATA on Model and returns the model

  OUTPUT FORMAT:
  DATA :== [data, labels]


  '''
  # Unpacking all the data
  new_data = NEW_DATA[0]
  new_labels = NEW_DATA[1]

  rehearsal_data = REHEARSAL_DATA[0]
  rehearsal_labels = REHEARSAL_DATA[1]

  old_test_data = OLD_TEST_DATA[0]
  old_test_labels = OLD_TEST_DATA[1]

  new_test_data = NEW_TEST_DATA[0]
  new_test_labels = NEW_TEST_DATA[1]

  accuracy_with_genetic_rehearsal=[]
  learning_with_genetic_rehearsal=[]

  for i in range(epochs):

    model.fit(new_data,new_labels,batch_size=100,epochs=1,verbose=1)
    model.fit(rehearsal_data,rehearsal_labels,batch_size=1000,epochs=1,verbose=1)

    score = model.evaluate(old_test_data,old_test_labels)

    accuracy_with_genetic_rehearsal.append(score[1])

    learning_score = model.evaluate(new_test_data,new_test_labels)
    print("Epoch Number: %d, Retention: %f, Learning: %f " %(i,score[1],learning_score[1]))
    learning_with_genetic_rehearsal.append(learning_score[1])

  return [accuracy_with_genetic_rehearsal, learning_with_genetic_rehearsal]
