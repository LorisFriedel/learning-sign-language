# ML

The purpose of this project is to create a real time sign language recognition software, using machine learning (powered by OpenCV).

## Version 1
From the OpenCV examples, we created a first version of a face detection software that loop on detecting the face in a given image, using pre-built configuration file from existing machine learning data.
Firstly, we increased the minimum rectangle size for the face detection. 
Using a 720p webcam, we could easily tell that a rectangle of 30x30 will most likely not contain a face if we are close to the camera. 
So we set it to 160x160 and thus reduced the detection time by 10. We can tweak it regarding our distance with the webcam. For us a value between 120 and 160 will fit our needs.
If we wanted to, we could have created a function that auto-adjust this value for us.

Then, we cropped the image in which the detection is performed using the last result of it, reducing time consumption by 4.

The next step was to use CamShift to track the face using colors (histograms) instead of detecting the face each frame.
We did that not to improve detection time but to be able to track the skin color to achieve the next step: track the hand.
(We have many ideas to improve detection quality, for example, removing the background (fixed pixel) or adjusting s_min, v_min, v_max and threshold automatically in real time)

## Version 2
After a bunch of minor and major improvements of the pure camshift program, we built the hand detection.
The algorithm is as follow:
 + Track the face on the frame (using previously built part of the program)
 + Remove the vertical area where the face is from the backprojection (using a mask)
 + Use camshift OpenCV function again on the modified backprojection
The result of this algorithm allows us to locate the hand (or the biggest object with the skin color in the image).

Using the hand location, we save the original hand image and the resized hand backprojection (16x16 in our  case).
The image is stored as a .png file and backprojection matrix as an .yml standardized file.

Next step was to create a class capable of learning through a data set of backprojection matrix (more generally a one line matrix, its length can be everything but every file data must have the same).
Using the OpenCV example, we created a multiple layer perceptron (MLP) that could learn to detect hand sign.
The MLPHand class can learn from any set of given data, but its output is only 26 indexes (that we could convert to letters, from a to z).
It is also possible to choose the number of hidden layer and the number of neuron per layer.
The generated model can be tested using our program.

To test the program, we saved 745 different images (and backproj) of sign from 'a' to 'f'.
After training a MLPHand with 2 layers and 128 neurons on each over 665 randomly picked backprojection data,
we tested it on the 80 remaining backproj data and obtained more 80% of recognition success.

The next step is to enlarge our data set to increase the success rate of our MLP.
We will also use a different data descriptor (using OpenCV HOG descriptor) to drastically improve the MLP performance.

## How to

/!\ We suggest to run the help command on each program to know what we can do with it, and how. /!\

Every run_*.sh script can be executed with the argument -h (or --help) to display help about available args and how to use the program.

+ Execute clean.sh to clean the project build
+ Execute build.sh to build every executable of the project
+ Execute run_*.sh args... to run the desired program (also build the project before execution if the executable is not present)
    
Author: Loris Friedel
