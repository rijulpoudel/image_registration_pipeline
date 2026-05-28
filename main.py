import sys
import os
sys.path.insert(0, 'src')
# telling python to look inside src/ when importing all the step files

import matplotlib
matplotlib.use('Agg')
# using Agg backend so matplotlib saves to file instead of trying to open a window
# this is needed on mac to avoid the non-interactive window error

from step1_load import load_images, display_images
from step2_detect import detect_keypoints, visualize_keypoints
from step3_match import match_keypoints, visualize_matches
from step4_homography import compute_homography, visualize_inliers
from step5_warp import warp_and_align, visualize_result


def run_pipeline(img1_path, img2_path):
    print("=" * 50)
    print("IMAGE REGISTRATION PIPELINE")
    print("=" * 50)

    # step 1 - loading both images and converting to grayscale
    print("\n[STEP 1] loading images...")
    img1c, img2c, img1g, img2g = load_images(img1_path, img2_path)
    display_images(img1c, img2c)

    # step 2 - detecting keypoints using ORB (interesting points in each image)
    print("\n[STEP 2] detecting keypoints with ORB...")
    kp1, desc1, kp2, desc2 = detect_keypoints(img1g, img2g, n_features=5000)
    visualize_keypoints(img1c, img2c, kp1, kp2)

    # step 3 - matching keypoints between the two images using brute force + lowe's ratio test
    print("\n[STEP 3] matching keypoints...")
    good_matches = match_keypoints(desc1, desc2, ratio_threshold=0.85)
    visualize_matches(img1c, img2c, kp1, kp2, good_matches)

    # step 4 - computing the homography matrix using RANSAC to filter out wrong matches
    print("\n[STEP 4] computing homography with RANSAC...")
    H, mask, inlier_matches = compute_homography(kp1, kp2, good_matches)
    visualize_inliers(img1c, img2c, kp1, kp2, inlier_matches)

    # step 5 - warping image 2 into image 1's coordinate frame using H
    print("\n[STEP 5] warping and aligning...")
    result, warped = warp_and_align(img1c, img2c, H)
    visualize_result(img1c, warped, result)

    print("\n[DONE] pipeline complete. all outputs saved to output/")


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    # making sure the output folder exists before saving anything
    run_pipeline("images/img1.jpg", "images/img2.jpg")