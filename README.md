# Genetic-Pseudo-Rehearsal
The repository is an official implementation of the paper  Pseudo Rehearsal using non-photo realistic images. <a href="https://arxiv.org/pdf/2004.13414.pdf"> Link to paper.</a> The work has been accepted at 25th International Conference on Pattern Recognition (ICPR) 2020. <a href="https://www.micc.unifi.it/icpr2020/"> Conference link.</a>

There are three implementations of the work. 
1. <b>MNIST_Fashion_demo.ipynb:</b> The entire code has been implemented in the notebook and was demonstrated on MNIST Fashion dataset by generating synthetic data for it and training a fresh neural network on it. Please refer to the paper for full details about the experiment. The .ipynb file can be readily used in Google colab environment.
2. <b>Genetic_Rehearsal.py:</b> The user can make the necessary function calls to implement the code. All the code for generating synthetic data and required supporting functions are implemented in this file.
3. <b>API:</b> To extend the availability of the algorithm to researchers who are using frameworks other than Tensorflow, we deployed the algorithm on a server as a web service. The code has been implemented as a web service using Python Flask and can be accessed via the following URL. (TO BE ADDED) Instructions for the usage of the API are here(TO BE ADDED).

Please find the link to the companion paper for our paper which describes the implementation in detail here. (TO BE ADDED)

# Intructions to run the .ipynb notebook:

Step 1: Upload the "Genetic_rehearsal.py" file to your working directory. If you are using google colab, upload it to runtime or mount it using your Google Drive.<br>
Step 2: Make sure that you are using Tensorflow 1.x version to run the file.<br>
Step 3: Run the .ipynb file.

Note:: The code to generate the synthetic data is in the .ipynb notebook itself. "Genetic_rehearsal.py" is being imported for some supporting functions.

# Instructions to run the .py file

TO BE ADDED SOON

# Instructions to use the API.

TO BE ADDED SOON.


For any queries please contact: cs18s506@iittp.ac.in or bhaskersuri@gmail.com
