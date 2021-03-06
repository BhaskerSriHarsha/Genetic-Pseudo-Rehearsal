# Genetic-Pseudo-Rehearsal
The repository is an official implementation of the paper  Pseudo Rehearsal using non-photo realistic images. <a href="https://arxiv.org/pdf/2004.13414.pdf"> Link to paper.</a> The work has been accepted at 25th International Conference on Pattern Recognition(ICPR), 2020, Milan, Italy.

To increase the availability of the work to a wider audience, the codes have been implemented in 3 formats. 
1. The first implementation is in a .ipynb notebook. The entire code has been implemented in the notebook and can be readily run as a Google colab notebook. These notebooks can be found in the <b>Jupyter notebooks</b> folder. The notebooks can be used to directly replicate the results for MNIST Fashion and CIFAR10 in Figure 3 of the main paper. They can be used to implement the algorithm on SVHN and MNIST Digits datasets as well.
2. The second implementation is a .py file. The code to generate the synthetic data has been implemented as a <i><b>function call</b></i>. Users can download the .py file to their working directory and <i><b>import</b></i> the function. An .ipynb notebook named (<b>Python_File_test_rig.ipynb</b>) has been added in the folder to demonstrate the usage of the .py file and the import function. <i>requriements.txt</i> file has been provided as well that can be used to setup the virtual environment.
3. To extend support to users using framework other than Tensorflow, an API version of the algorithm was implemented. The code can be deployed on a local server and can be accessed by any neural network implemented in any language.

# Intructions to run the Jupyter notebook:

Step 1: Upload the "Genetic_rehearsal.py" file to your working directory. If you are using google colab, upload it to runtime or mount it using your Google Drive.<br>
Step 2: Make sure that you are using Tensorflow 1.x version to run the file.<br>
Step 3: Run the .ipynb file.

# Instructions to run the .py file

The entire code to generate the synthetic data has been implemented as a function call in the file <b>Genetic_Rehearsal.py</b>. The file also contains other supporting code blocks to implement various selection mechanisms in genetic algorithms. To use the function call, download the file titled <b>Genetic_Rehearsal.py</b> from the folder <b>.py files</b> to your working directory. Create a virtual environment with the required requirments that can be found in the <b>requirments.txt</b> file.

# Instructions to use the API.

This implementation is intended for users who built neural networks using any framework other than Tensorflow. The algorithm can be deployed on any local server and synthetic data for the target neural network can be generated by interacting with that server. The target neural network needs to access the server using HTTP GET and POST requests. The algorithm described in <b> CLIENT SIDE ALGORITHM</b> should be used to generate the synthetic data on the server. If you are using any Pythonian framework like PyTorch or Sklearn to implement your neural network, a sample python template(<i>client.py</i>) is already provided in the folder titled <b> API</b>. 

<center><h3>CLIENT SIDE ALGORITHM</h3></center>
<hr>      
      
      status_flag=0
      target_labels="1,2,3"
      POST('/', data=target_labels, timeout=1)
      status_flag=GET('/training')
      while status_flag == 1:
        images_flag = GET("/flag")
        if images_flag == 1:
          images = GET('/images')
          predictions = model(images)
          POST('/predictions',data=predictions)
          POST('/reset_flag',data=0)
          POST('/ready',data=1)
        status_flag=GET('/training')
      
      synthetic_data=GET('/synthetic_data')
      


For any queries please contact: cs18s506@iittp.ac.in or bhaskersuri@gmail.com
