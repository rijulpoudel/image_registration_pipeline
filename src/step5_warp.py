import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def warp_and_align(img1_color, img2_color, H):
    h1, w1 = img1_color.shape[:2]
    # getting height and width of image 1
    # [:2] slices the first two elements of shape, ignoring the channel count (3)

    h2, w2 = img2_color.shape[:2] # same for image 2

    output_width = w1 + w2 # canvas needs to be wide enough to hold both images side by side
    output_height = max(h1, h2) # canvas needs to be tall enough for whichever image is taller

    warped_img2 = cv2.warpPerspective(img2_color, H, (output_width, output_height))
    # this is where the magic happens
    # applies H to transform every single pixel of img2 to its new location
    # think of it as physically sliding and rotating img2 until it lines up with img1
    # pixels that map outside the canvas boundaries just become black

    result = warped_img2.copy() # start with warped img2 as our base canvas
    result[0:h1, 0:w1] = img1_color
    # places image 1 on the left side of the canvas
    # array slicing: rows 0 to h1, columns 0 to w1
    # this overwrites that region of the canvas with the original img1

    cv2.imwrite("output/result_final.jpg", result) # saving the final aligned composite to disk
    return result, warped_img2


def visualize_result(img1_color, warped_img2, result):
    # shows three images side by side so we can compare:
    # original img1, warped img2, and the final stitched composite
    fig, axes = plt.subplots(1, 3, figsize=(18, 5)) # three subplots in one row, wide figure to fit all three

    axes[0].imshow(cv2.cvtColor(img1_color, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Image 1 (Base)", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(warped_img2, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Image 2 Warped to Image 1 Frame", fontsize=12, fontweight='bold')
    # warped img2 should look like img1's perspective now
    # the content should roughly line up with img1 in the overlap region
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Final Aligned Composite", fontsize=12, fontweight='bold')
    # this is the final result - both images stitched together into one wide image
    # the overlap region should look seamless or close to it
    axes[2].axis('off')

    plt.suptitle("Step 5: Warp and Alignment Result", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig("output/step5_result.jpg", dpi=150, bbox_inches='tight')
    print("saved: output/step5_result.jpg")
    print("saved: output/result_final.jpg")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, 'src') # tells python where to find the other step files
    from step1_load import load_images
    from step2_detect import detect_keypoints
    from step3_match import match_keypoints
    from step4_homography import compute_homography
    img1c, img2c, img1g, img2g = load_images("images/img1.jpg", "images/img2.jpg")
    kp1, desc1, kp2, desc2 = detect_keypoints(img1g, img2g)
    good = match_keypoints(desc1, desc2)
    H, mask, inliers = compute_homography(kp1, kp2, good)
    result, warped = warp_and_align(img1c, img2c, H)
    visualize_result(img1c, warped, result)