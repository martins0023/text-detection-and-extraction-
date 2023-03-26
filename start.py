import os
from PIL import Image #import python library that processes your image

def resize_img(): #create a function
    
    img = Image.open("image.png") #process the image 
    
    if img.height > 300 or img.width > 300: #get the height and width of the image
        output_size = (900, 900) #crop the image to the output size
        img.thumbnail(output_size) 
        ext = ['.jpeg', '.png', '.jpg'] #create list of extensions to save as 
        for extension in ext: #loop over the list 
            img.save(f"image_resize{extension}") #save the image with a new name with the list of extensions added after preocessing

    os.system('python3 test.py')        
        
resize_img()