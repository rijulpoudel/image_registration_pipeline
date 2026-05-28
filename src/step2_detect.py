import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
[]
def detect_keypoints(img1_gray, img2_gray, n_features=2000):
    orb = cv2.ORB_create(nfeatures=n_features) 
    # the max number of keypoints it is allowed to find per image is n_features
    #the higher the number of keypoints, more the potential matches but slower

    kp1, desc1 = orb.detectAndCompute(img1_gray, None)
    #it does two things at once:
    #detect - finds all the keypoints in the image. 
    #compute - calculates the descriptor for each keypoint
    #kp1 gets the list of keypoint objects(locations)
    #desc1 gets the array of descriptors(fingerprints)
    #the None means no mask - use the entire image, not just a specific region. 
    kp2, desc2 = orb.detectAndCompute(img2_gray, None)

    print(f"keypoints found in img1: {len(kp1)}") #sanity check to see how many keypoints ORB found
    print(f"keypoints found in img2: {len(kp2)}") #this should be around 500-2000-ish

    print(f"descriptor shape for img1: {desc1.shape}")
    #prints the shape of the descriptor array
    #it should return somthing like (1847, 32)
    #the first number is how many keypoints were found
    #the second number is always 32 since it's a 32 bytes per descriptor

    return kp1, desc1, kp2, desc2


def visualize_keypoints(img1_color, img2_color, kp1, kp2): #function to draw and display the keypoints
    img1_kp = cv2.drawKeypoints(
        img1_color, kp1, None, #img1_color is the sources of the image to draw on, kp1 is the list of keypoints, Noen means create a new output image rather than drawing on the original
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS #flag makes the circles show size and orientation, not just a plain dot
    )#this draws circle at every keypoint location.

    img2_kp = cv2.drawKeypoints(
        img2_color, kp2, None, 
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    img1_kp_rgb = cv2.cvtColor(img1_kp, cv2.COLOR_BGR2RGB) #BGR to RGB conversion since open cv draws the keypont circles in BGR format.
    img2_kp_rgb = cv2.cvtColor(img2_kp, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))#creating the two plots side by side as step 1, axes is an array of the two boxes

    axes[0].imshow(img1_kp_rgb) #places img1_kp_rgb plot inside the left box
    axes[0].set_title(f"Image 1 - {len(kp1)} Keypoints", fontsize=13)
    axes[0].axis('off')
    
    axes[1].imshow(img2_kp_rgb)
    axes[1].set_title(f"Image 2 — {len(kp2)} Keypoints", fontsize=13)
    axes[1].axis('off')

    plt.suptitle("Step 2: ORB Keypoint Detection", fontsize=16, fontweight='bold') #supertitle(above both the plots)
    plt.tight_layout()
    plt.savefig("output/step2_keypoints.jpg", dpi=150, bbox_inches='tight')
    print("saved: output/step2_keypoints.jpg")



if __name__ == "__main__":
    import sys
    sys.path.insert(0, 'src') # tells python to look inside src/ when importing. without this, from step1_load import load_images would fail because Python wouldn't know where to find it. 
    from step1_load import load_images
    img1c, img2c, img1g, img2g = load_images("images/img1.jpg", "images/img2.jpg")
    kp1, desc1, kp2, desc2 = detect_keypoints(img1g, img2g)
    visualize_keypoints(img1c, img2c, kp1, kp2)
