## Detection of Aggregate Reticulocyte in Feline Blood Smears

I applied a Convolutional Neural Network (CNN) architecture to solve an object detection task to identify aggregate reticulocytes in blood smears of feline patients. 

***
## Introduction 

Financial constraints are among the most prevalent reasons that prevent a pet owner from seeking veterinary care for their pet. As reported by Betterpet[1], the expenses associated with an emergency visit alone can skyrocket, reaching upwards of $13,000. This substantial financial burden is due to the cost of diagnostic imaging tests such as CT scan, radiographs, and ultrasounds that require a board certified specialist to perform and interpret. Not only are these cost out-of-pocket, they usually require the pet to be anesthetized, raising the cost even more.  Recognizing these financial challenges, there is a growing motivation to explore the integration of machine learning technologies to potentially alleviate these financial constraints in veterinary care. 

Advancements in machine learning in human medicine are much greater than that of veterinary care when searching on Google Scholar. For example, one paper developed a model to analyze chest CT scan images to help diagnose COVID-19. Among the research, the most interesting is the use of machine learning and deep learning in oncology. Searches for similar research in veterinary medicine yield less robust results. This discrepancy is likely due to the limited data collected and the ethical dilemmas faced. Notably, the potential for a misdiagnosis may inadvertently lead to owners to opt for euthanisa since the cost of treating the issue may be overwhelming. Therefore, there is a need for more research and development of machine learning technologies tailored for veterinary medicine that enhance diagnostic accuracy. 

One example of utilizing machine learning is the diagnosis of anemia in feline patients. The first step would be conducting a complete blood cell count (CBC) on a blood analyzer. If the pet is found to have a low red blood cell (RBC) count, a blood smear will have to be stained to count the presence of immature red blood cells, known as reticulocytes. This will let the veterinarian know if the bone marrow is responding appropriately [2]. The challenge lies in distinguishing between aggregate reticulocyte and punctate reticulocyte within the blood smear. Only the aggregate reticulocyte should be counted because punctate reticulocytes, having been in the blood for several days, do not accurately reflect bone marrow response [3]. As Figure 1 shows, they have similar structure, except aggregate reticulocyte have more clumps a blue-stained granules. 

![Image of aggregate and punctate reticulocyte]([https://raw.githubusercontent.com/username/repository/main/path/to/image.jpg](https://eclinpath.com/wp-content/uploads/aggregate-punctate.png))
Figure 1: Image of aggregate and punctate reticulocyte [4]

The goal of this project was to use supervised learning techniques, specifically a CNN architecture, to train a model to locate an aggregate reticulocyte within a 300 x 300 image of a stained blood smear. The blood smear contained multiple classes and objects, but for this project, I only looked to locate one object of one class per image. Regression was used to predict the spatial coordinate (boundary boxes) of the aggregate reticulocytes within the given image. 

The model was able to predict the boundary boxes with a final Intersection over Union score of 0.61 for the training set and a score of 0.62 for the training set and 0.49 for the testing set.


## Data 

The dataset for this project was obtained from Kaggle that consisted of two files, one containing images as JPEG images and the other as XML files containing bounding box information in Pascal Visual Object Classes (VOC) format as shown below. This format is commonly used for computer vision tasks since it annotates objects within images. 

***INSERT IMAGE***
Figure 2: Sample XML file showing Pascal VOC format

Since I worked on a Google Collab workspace, I was able to upload the dataset to my GoogleDrive into two folders (labels and images). As mention, since the XML files were all in Pascal VOC format, I was able to easily parse through all the files to extract the bounding box coordinates of only the objects labeled “aggregate reticulocyte” using xml.etree.ElementTree. I also extracted the image file name to confirm the bounding box information was being extracted correctly. These lists of coordinates were added to another data list that was then converted into a DataFrame to handle the data more efficiently. 

	
<object>
		<name>aggregate reticulocyte</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
	<bndbox>
		<xmin>138</xmin>
		<ymin>229</ymin>
		<xmax>185</xmax>
		<ymax>277</ymax>
	</bndbox>
</object>

*Figure 3: Comparing the information in the dataframe to the corresponding 001173.xml file*

Now, we have 1001 images, considering the division based on bounding box information. Ultimately, this means that a single image might contribute to both the training or testing sets, introducing bias to the results. Another approach would be to only select the first unique instance in the list of data. However, this would involve data augmentation to ensure diversity, but this can introduce noise. 

The data frame was then split into a training and testing dataframe before preprocessing using the sklearn.model_selection.train_test_split()method. 

Now that the data is split into training and test data, we need to convert the image and bounding box information need to be converted into NumPy arrays. First, we needed to load the input (images). Since we are loading the images using OpenCV, the default color representation is BGR, but Matplotlib operates in RGB. So if I wanted to visualize the data, it made sense to convert into RGB. For each row in the dataframe, a function extracted the “file name”, uploaded the image, and stored the pixel information as an array. Extracting bounding box information was straightforward as it was already organized into rows in the dataframe. 


Since we already split the main dataframe into test_df and train_df, we need the input and output arrays for the training and testing dataframes. This results in our x_train, y_train, x_test, and y_test as NumPy arrays. The x_test and x_train were shaped as (number of samples, image width, image height, and channels). The y_test and y_train were shaped as (number of samples, number of coordinates). 

 x_train shape: (800, 300, 300, 3)
y_train shape: (800, 4)

x_train shape: (201, 300, 300, 3)
x_train shape: (201, 4)


To normalize the images, I divide the values in x_train and x_test by 255 to get a range from 0 to 1. I then visualized some of the sample to make sure the information is being stored properly. This was done for both the x_train and the x_test.  


Figure X clearly has only 1 object while Figure 2 contains more than 1. 

Before building the model, it is important to also scale the outputs since the images normalized. y_train = y_train / [image_width, image_height, image_width, image_height]
y_test = y_test / [image_width, image_height, image_width, image_height]

