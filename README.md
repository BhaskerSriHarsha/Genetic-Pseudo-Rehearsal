# Genetic-Pseudo-Rehearsal
The repository is an official implementation of the paper  Pseudo Rehearsal using non-photo realistic images. <a href="https://arxiv.org/pdf/2004.13414.pdf"> Link to paper.</a>

There are three implementations of the work. 
1. The first implementation is in a .ipynb notebook. The entire code has been implemented in the notebook and can be readily run as a Google colab notebook.
2. The second implementation is a .py file through which the user can make the necessary function calls to implement the code.
3. The code has been implemented as an API and can be accessed via the following URL

# Intructions to run the .ipynb notebook:

Step 1: Upload the "Genetic_rehearsal.py" file to your working directory. If you are using google colab, upload it to runtime or mount it using your Google Drive.<br>
Step 2: Make sure that you are using Tensorflow 1.x version to run the file.<br>
Step 3: Run the .ipynb file.

# Instructions to run the .py file

TO BE ADDED SOON

# Instructions to use the API.

The work has been deployed on a server and can be accessed using HTTP methods. 

<center><h3>ALGORITHM</h3></center>
<hr>      
      
      training_flag=0
      target_labels="1,2,3"
      POST('home_page', data=target_labels, timeout=1)
      trianing_flg=GET('home/training')
      while training_flag == 1:
        flag = GET("home/flag")
        if flag == 1:
          images = GET('home/images')
          predictions = model(images)
          POST('home/predictions',data=predictions)
          POST('home/reset_flag',data=0)
          POST('home/ready',data=1)
        training_flag=GET('home/training')
        synthetic_data=GET('home/synthetic_data')
      


For any queries please contact: cs18s506@iittp.ac.in or bhaskersuri@gmail.com
