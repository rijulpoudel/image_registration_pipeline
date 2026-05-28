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

    img1_color = cv2.resize(img1_color, (1080, 1440))  # <-- add here
    img2_color = cv2.resize(img2_color, (1080, 1440))
    img1_gray = cv2.resize(img1_gray, (1080, 1440))
    img2_gray = cv2.resize(img2_gray, (1080, 1440))


    return img1_color, img2_color, img1_gray, img2_gray




def display_images(img1_color, img2_color):
    img1_rgb = cv2.cvtColor(img1_color, cv2.COLOR_BGR2RGB) 
    #since matplotlib need rgb, we convert bgr to rgb
    img2_rgb = cv2.cvtColor(img2_color, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    #subplots creates a grid. 1 represents rows, so it will have 1 row and 2 columns (basically for the 2 images) 
    #figsize creates the dimensions of the object 12 x 5
    #fig is the overall figure container and axes is an array of two plot areas, one for each image. Can access them as axes[0] and axes[1]

    axes[0].imshow(img1_rgb) #places the img1_rgb inside the left box. nothing displays yet since we havent don't plt.show() yet.
    axes[0].set_title("Image 1 (Base)", fontsize=14) #adds ttile above left subplot
    axes[0].axis('off')#hides the x and y axis ticks and numbers around the image. since images aren't graphs, we don't need it.
    axes[1].imshow(img2_rgb)
    axes[1].set_title("Image 2 (To Align)", fontsize=14)
    axes[1].axis('off')


    plt.suptitle("Step 1: Loaded Images", fontsize=16, fontweight='bold') #it adds one big title above both subplots combined

    plt.tight_layout() #adjusts the spacing between subplots so nothing overlaps. fixes the spacing issues

    plt.savefig("output/step1_loaded.jpg", dpi=150, bbox_inches='tight') #save to output folder as step1_loaded.jpg, dpi 150 and bbox_inches crops any extra whitespace 

    plt.show()

    print("saved: output/step1_loaded.jpg")



if __name__ == "__main__":
    img1c, img2c, img1g, img2g = load_images("images/img1.jpg", "images/img2.jpg")
    display_images(img1c, img2c)






