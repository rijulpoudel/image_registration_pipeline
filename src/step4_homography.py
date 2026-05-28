import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def compute_homography(kp1, kp2, good_matches, ransac_threshold=5.0):
    if len(good_matches) < 4:
        raise ValueError(f"not enough matches: {len(good_matches)} (need at least 4)")
        # homography needs a minimum of 4 point pairs to solve
        # if we have fewer, we raise a clear error instead of crashing weirdly later

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    # extracting the (x, y) coordinates of matched keypoints in image 1
    # m.queryIdx is the index of the keypoint in kp1
    # .pt gives the (x, y) tuple of that keypoint
    # reshape(-1, 1, 2) is the exact shape findHomography expects - (N, 1, 2)

    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    # same thing but for image 2
    # m.trainIdx is the index of the matched keypoint in kp2

    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, ransac_threshold)
    # this is the big one - computes the 3x3 homography matrix H
    # pts2 is first (source - image 2), pts1 is second (destination - image 1)
    # this gives us H that transforms image 2 into image 1's coordinate frame
    # cv2.RANSAC tells it to use RANSAC to automatically reject wrong matches
    # ransac_threshold is the max allowed pixel error to count a match as an inlier
    # mask is an array of 1s and 0s - 1 means inlier match, 0 means outlier rejected by RANSAC

    n_inliers = int(mask.sum()) # counts inlier matches - mask.sum() adds up all the 1s
    n_outliers = len(good_matches) - n_inliers

    print(f"inliers: {n_inliers} / outliers rejected: {n_outliers}")
    print(f"inlier ratio: {n_inliers/len(good_matches)*100:.1f}%")
    # inlier ratio above 60% means a reliable homography
    # if its below 30% something is wrong - images might not overlap enough

    print(f"homography matrix H:\n{H}")
    # H is a 3x3 matrix of floats
    # it encodes rotation, translation, scale and perspective all in one

    inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i]]
    # filtering good_matches to only keep the inliers
    # used for clean visualization in the next function

    return H, mask, inlier_matches


def visualize_inliers(img1_color, img2_color, kp1, kp2, inlier_matches):
    # shows only the clean inlier matches after RANSAC filtered out the wrong ones
    match_img = cv2.drawMatches(
        img1_color, kp1,
        img2_color, kp2,
        inlier_matches[:50], None, # only drawing first 50 so it doesnt get too cluttered
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS # hides keypoints with no match
    )

    match_rgb = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB) # BGR to RGB for matplotlib
    match_rgb = cv2.resize(match_rgb, (1600, 500)) # resizing so lines are visible

    plt.figure(figsize=(16, 6))
    plt.imshow(match_rgb)
    plt.title(f"Step 4: Inlier Matches After RANSAC — {len(inlier_matches)} clean matches", fontsize=14)
    # should look much cleaner than step 3 - mostly parallel lines, no crazy crossed ones
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("output/step4_inliers.jpg", dpi=150, bbox_inches='tight')
    print("saved: output/step4_inliers.jpg")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, 'src') # tells python where to find the other step files
    from step1_load import load_images
    from step2_detect import detect_keypoints
    from step3_match import match_keypoints
    img1c, img2c, img1g, img2g = load_images("images/img1.jpg", "images/img2.jpg")
    kp1, desc1, kp2, desc2 = detect_keypoints(img1g, img2g)
    good = match_keypoints(desc1, desc2)
    H, mask, inliers = compute_homography(kp1, kp2, good)
    visualize_inliers(img1c, img2c, kp1, kp2, inliers)