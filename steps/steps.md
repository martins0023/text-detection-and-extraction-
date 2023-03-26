**Startup**
* Click on the icon "text detection and extraction" to start or power on the program.

**steps**
1. to detect and extract texts from an image, place or copy the image text to be extracted in the parent working directory, then click on the icon "text detection and extraction" to start or power on the program, in the top left corner of the gui interface,click on the button "Get started now" and wait a few minutes for the processing of the image to be completed, once it is completed there will be a text that says "succeeded /n check directories". To check for the file ouput in the directories; in the home folder directory, click on 'PROCESSED', followed by the 'TEXT DETECTED' folder and open the file "text_detected.txt" in the folder.
'*processed/text_detected/text_detected.txt*'

**NOTE** 
    Operating system - This project was fully tested and developed on Linux 
    Programming language - Python3
    
    Source code editor - Visual Studio code
    System Requirement - 4gbram, 500hd

## passing in the image to be used/test
put the image file in the base working directory, rename the file as "image" in file format ".png" or make the changes directly in the 'test.py' file, on line '427' 
where **image = cv2.imread('new.png')**

## step 1
Installation of dependencies used if it's not available on the host

## step 2 
Import the modules to be used

## step 3
Importing of the datasets to be uses. The datasets used for this project is COCO and it is available in a zip format in the home directory. Once the dataset is imported, it unzips the file and prepares it. JSON module is used to prepare the unzipped datasets stored in a 'JSON' format.

## step 4
registering the data, from detectron2 we import MetadataCatalog and DatasetCatalog. The DatasetCatalog is used to register the imported dataset which is COCO in this case, and the MetadataCatalog is used to detect the type of language the text on the image is, and train it

## step 5
in the first phase of visualizing the image, we use a for loop to select random datasets from the prepared data

## step 6 
traning and eveluation of the data 

## step 7 
performing binarization and skew correction of the image

## step 8
detecting key points characters on the images

## step 9
performing preprocessing of the image, to get the text extracted from the image processed, then a function to draw boxes on text area detected is parsed, then the text detected on the image is extracted and stored in a file txt in;
    /processed/text_detected/file-format.txt

**files-processed**
all images processed are stored in /processed/images
all text detected are stored in /processed/text_detected

**images-to-be-processed**
images to be processed are stored in /images

