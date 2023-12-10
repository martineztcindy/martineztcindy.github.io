## Detection of Aggregate Reticulocyte in Feline Blood Smears

I applied a Convolutional Neural Network (CNN) architecture to solve an object detection task to identify aggregate reticulocytes in blood smears of feline patients. 

***
## Introduction 

Financial constraints are among the most prevalent reasons that prevent a pet owner from seeking veterinary care for their pet. As reported by Betterpet[1], the expenses associated with an emergency visit alone can skyrocket, reaching upwards of $13,000. This substantial financial burden is due to the cost of diagnostic imaging tests such as CT scan, radiographs, and ultrasounds that require a board certified specialist to perform and interpret. Not only are these cost out-of-pocket, they usually require the pet to be anesthetized, raising the cost even more.  Recognizing these financial challenges, there is a growing motivation to explore the integration of machine learning technologies to potentially alleviate these financial constraints in veterinary care. 

Advancements in machine learning in human medicine are much greater than that of veterinary care when searching on Google Scholar. For example, one paper developed a model to analyze chest CT scan images to help diagnose COVID-19. Among the research, the most interesting is the use of machine learning and deep learning in oncology. Searches for similar research in veterinary medicine yield less robust results. This discrepancy is likely due to the limited data collected and the ethical dilemmas faced. Notably, the potential for a misdiagnosis may inadvertently lead to owners to opt for euthanisa since the cost of treating the issue may be overwhelming. Therefore, there is a need for more research and development of machine learning technologies tailored for veterinary medicine that enhance diagnostic accuracy. 

One example of utilizing machine learning is the diagnosis of anemia in feline patients. The first step would be conducting a complete blood cell count (CBC) on a blood analyzer. If the pet is found to have a low red blood cell (RBC) count, a blood smear will have to be stained to count the presence of immature red blood cells, known as reticulocytes. This will let the veterinarian know if the bone marrow is responding appropriately [2]. The challenge lies in distinguishing between aggregate reticulocyte and punctate reticulocyte within the blood smear. Only the aggregate reticulocyte should be counted because punctate reticulocytes, having been in the blood for several days, do not accurately reflect bone marrow response [3]. As Figure 1 shows, they have similar structure, except aggregate reticulocyte have more clumps a blue-stained granules. 


<img src = "https://github.com/martineztcindy/martineztcindy.github.io/blob/main/aggregate-punctate-300x179.png?raw=true.png" alt = "aggpunt" width = 300>

*Figure 1: Image of aggregate and punctate reticulocyte [4]*

The goal of this project was to use supervised learning techniques, specifically a CNN architecture, to train a model to locate an aggregate reticulocyte within a 300 x 300 image of a stained blood smear. The blood smear contained multiple classes and objects, but for this project, I only looked to locate one object of one class per image. Regression was used to predict the spatial coordinate (boundary boxes) of the aggregate reticulocytes within the given image. 

The model was able to predict the bounding boxes with a final Intersection over Union score of 0.52 for the training set and 0.53 for the testing set. The final mean average error (MAE) was 0.074 and 0.073 for the training and testing set, respectively.


## Data 

The dataset for this project was obtained from Kaggle that consisted of two files, one containing images as JPEG images and the other as XML files containing bounding box information in Pascal Visual Object Classes (VOC) format as shown below. This format is commonly used for computer vision tasks since it annotates objects within images. 

Since I worked on a Google Collab workspace, I was able to upload the dataset to my GoogleDrive into two folders (labels and images). As mention, since the XML files were all in Pascal VOC format, I was able to easily parse through all the files to extract the bounding box coordinates of only the objects labeled “aggregate reticulocyte” using xml.etree.ElementTree. I also extracted the image file name to confirm the bounding box information was being extracted correctly. These lists of coordinates were added to another data list that was then converted into a DataFrame to handle the data more efficiently. 

![dataframe example](https://github.com/martineztcindy/martineztcindy.github.io/blob/main/Capture.PNG?raw=true)
![XML file](https://github.com/martineztcindy/martineztcindy.github.io/blob/main/Capture2.PNG?raw=true)
*Figure 2: Comparing the information in the dataframe to the corresponding 001173.xml file*

Now, we have 1001 images, considering the division based on bounding box information. Ultimately, this means that a single image might contribute to both the training or testing sets, introducing bias to the results. Another approach would be to only select the first unique instance in the list of data. However, this would involve data augmentation to ensure diversity, but this can introduce noise. 

The data frame was then split into a training and testing dataframe before preprocessing using the sklearn.model_selection.train_test_split()method. 

Now that the data is split into training and test data, we need to convert the image and bounding box information need to be converted into NumPy arrays. First, we needed to load the input (images). Since we are loading the images using OpenCV, the default color representation is BGR, but Matplotlib operates in RGB. So if I wanted to visualize the data, it made sense to convert into RGB. For each row in the dataframe, a function extracted the “file name”, uploaded the image, and stored the pixel information as an array. Extracting bounding box information was straightforward as it was already organized into rows in the dataframe. 

Since we already split the main dataframe into test_df and train_df, we need the input and output arrays for the training and testing dataframes. This results in our x_train, y_train, x_test, and y_test as NumPy arrays. The x_test and x_train were shaped as (number of samples, image width, image height, and channels). The y_test and y_train were shaped as (number of samples, number of coordinates). 

	x_train shape: (800, 300, 300, 3)
	y_train shape: (800, 4)

	x_train shape: (201, 300, 300, 3)
	x_train shape: (201, 4)

To normalize the images, I divide the values in x_train and x_test by 255 to get a range from 0 to 1. I then visualized some of the sample to make sure the information is being stored properly. This was done for both the x_train and the x_test.  

<img src = "https://github.com/martineztcindy/martineztcindy.github.io/blob/main/image1.png?raw=true.png" alt = "img1" width = 300>
<img src = "https://github.com/martineztcindy/martineztcindy.github.io/blob/main/image2.png?raw=true.png" alt = "img1" width = 300>

*Figure 3: Image 2 clearly has only 1 object while Image 1 contains more than 1.*

Before building the model, it is important to also scale the outputs since the images normalized. 

``y_train = y_train / [image_width, image_height, image_width, image_height]
  y_test = y_test / [image_width, image_height, image_width, image_height]``

  ## Modeling

``model = tf.keras.Sequential([

    # First convolutional layer with 32 filters
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    layers.MaxPooling2D((2, 2)), # reduce the spatial dimension

    # Second convultional layer with 64 filters
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)), # reduce spatial dimension


    layers.Flatten(), # flatten features

    # connect layers
    layers.Dense(128, activation='relu'), # 128 neurons
    layers.Dense(64, activation='relu'), # 64 neurons
    layers.Dense(32, activation='relu'), # 32 neurons

    layers.Dense(4, activation='linear') # 4 nuerons and linear activation
]``

I chose to use tf.keras.Sequential()to construct my CNN network since it is suitable for object detection tasks that process features across multiple layers. The model is designed to handle 300 x 300 pixel images with the layers to help learn and predict bounding box coordinates for the target object, aggregate reticulocyte. 

The model consists of convolutional layers to capture the features (edges, textures, and patterns). The first convolutional layer uses 32 filters with a size of 3 x 3. It activates these features using the ‘relu’. I chose Rectified Linear Unit (ReLU) activation function since it had a lower loss rate than the sigmoid function. Max-pooling layer helps to shrink the spatial size of the feature. This is repeated again with 64 filters to help with feature extraction. The features are flattened in a single list. The first dense layer has 128 neurons, the second has 64, and the third has 32. The final output layer has four neurons, representing the four bounding box coordinates). This time, linear activation was selected because the output predicted numerical values for the location. 

Now we can compile the model with our training and testing data using the IoU metric. The IoU metric was selected because it is the best for assessing the accuracy of object detection and localization. It measures the overlap between predicted and the ground truth bounding boxes. I used the mean absolute error (MAE) for my loss function because it is well-suited for regression tasks, such as predicting bounding box coordinates. MAE calculates the average absolute difference between the predicted and actual values, helping localize the object target in the images. 

The model.fit() function was used to train the CNN with the training set for 10 epochs, a batch size set to 32, and validating using the test data. 

## Results
Figure 4 shows the plot for the MAE over epochs. Figure 5 shows the plot for the IoU over epochs. 
<img src = "https://github.com/martineztcindy/martineztcindy.github.io/blob/main/mae.png?raw=true.png" alt = "mae" width = 500>
<img src = "https://github.com/martineztcindy/martineztcindy.github.io/blob/main/iou.png?raw=true.png" alt = "iou" width = 500>
*Figures 4-5: MAE over epochs graph. IoU over epochs graph.*

Below are samples of images from the testing and training predictions that contain both the predicted and ground truth bounding boxes. 

<img src = "https://github.com/martineztcindy/martineztcindy.github.io/blob/main/trainpred2.png?raw=true.png" alt = "new1" width = 300>
<img src = "https://github.com/martineztcindy/martineztcindy.github.io/blob/main/trainpred3.png?raw=true.png" alt = "new2" width = 300>
<img src = "https://github.com/martineztcindy/martineztcindy.github.io/blob/main/predtest2.png?raw=true.png" alt = "new3" width = 300>
<img src = "https://github.com/martineztcindy/martineztcindy.github.io/blob/main/predtest1.png?raw=true.png" alt = "new4" width = 300>

*Figures 6-9: Sample images from testing and training predictions*

Finally, I tested images not previously shown by the model. Unfortunately, I do not have the annotations so I cannot assess the results. However, visually one can see the model’s capability in detecting aggregate reticulocytes. 

<img src = "https://github.com/martineztcindy/martineztcindy.github.io/blob/main/newimage1.png?raw=true.png" alt = "new5" width = 300>
<img src = "https://github.com/martineztcindy/martineztcindy.github.io/blob/main/newimage2.png?raw=true.png" alt = "new6" width = 300>
<img src = "https://github.com/martineztcindy/martineztcindy.github.io/blob/main/newimage4.png?raw=true.png" alt = "new7" width = 300>
<img src = "https://github.com/martineztcindy/martineztcindy.github.io/blob/main/newimage3.png?raw=true.png" alt = "new8" width = 300>

*Figures 10-3: Sample images from never-before-seen images and the prediction boundary box*

## Discussion
The model did not prove to be as robust for predicting bounding boxing, with an IoU score of 0.52 for the training set, 0.53 for the testing test. Typically, an IoU score above 0.5 indicates a strong predictive performance. We would expect a great increase in IoU score for the testing test, so the slight increase in the testing set’s IoU score could potentially be attributed to repeated instances of the same images, given the dataset division by bounding box rather than unique file names. Despite these challenges, the relatively low MAE score for both the testing and training sets suggest accurate boundary box regression. The most impressive part about the model’s performance was its ability to accurately detect aggregate reticulocytes in images it had never seen before. 

## Conclusion 
When I first started this project, the initial task was to build a multi-task, multi-object detector. This was because the original dataset contains images with multiple objects with three distinct classes. So, not only would this be a regression task, but also a classification task. However, the main obstacle I faced was the dynamic nature of the data. The variability in both input shapes and corresponding output structures need to be carefully considered when building a CNN. 

As mentioned, the reduction in sample size based on the bounding box affected the model’s performance. Despite these challenges, the fact that the model successfully located aggregate reticulocytes in images that it had not encountered before proved that with some fine-tuning, this model could be adapted to detect multiple images. Object detection is a field in veterinary medicine that requires further research, particularly in addressing challenges that could contribute to reducing the cost of diagnostic imaging. The necessity for more extensive datasets becomes apparent, as the model would thrive on variance.


## References 
[1]betterpet.com/emergency-vet-costs
[2]vcahospitals.com/know-your-pet/anemia-in-cats
[3]klimud.org/public/atlas/idrar/web/www.diaglab.vet.cornell.edu/clinpath/modules/rbcmorph/fretic.htm
[4]eclinpath.com/hematology/hemogram-basics/definitions-and-terms/aggregate-punctate


