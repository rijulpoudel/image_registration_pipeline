import cv2
import matplotlib.pyplot as plt


def load_images(path1, path2):
    img1_color = cv2.imread(path1) #reads an image file from disk and converts it into a numpy array
    #result stored in img1_color will be a 3d array shaped like (height, width, 3)
    img2_color = cv2.imread(path2)

    if img1_color is None:
        raise FileNotFoundError(f"could not load image: {path1}") #just checking if img1_color returns None silently on failure
    if img2_color is None:
        raise FileNotFoundError(f"could not load image: {path2}")
    
    img1_gray = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY) #cv2.cvtColor converts an image from one color space to another (BGR to Grayscale here)
    #ORB works on brighness values, not color so we use grayscale here. This is faster to process and color doesn't help find corners or grades. 
    img2_gray  =cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)


    print(f"img1 shape: {img1_color.shape}") #.shape returns dimension of the array as a tuple
    print(f"img2 shape: {img2_color.shape}")    

    return img1_color, img2_color, img1_gray, img2_gray


load_images('images/img1.jpg', 'images/img2.jpg')




