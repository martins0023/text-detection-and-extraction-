#crop any picture to your satisfaction with this python program and save to different filenames
#python scripting for computational science 
#by miracle oladapo

from PIL import Image #import python library that processes your image

def save(): #create a function
    img = Image.open("sample-image.jpg") #process the image 
    
    if img.height > 300 or img.width > 300: #get the height and width of the image
        output_size = (500, 450) #crop the image to the output size
        img.thumbnail(output_size) 
        ext = ['.jpeg', '.png', '.jpg'] #create list of extensions to save as 
        for extension in ext: #loop over the list 
            img.save(f"save{extension}") #save the image with a new name with the list of extensions added after preocessing
        
save()
