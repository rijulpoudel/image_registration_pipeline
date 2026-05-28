import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def match_keypoints(desc1, desc2, ratio_threshold=0.85): 
    #function takes descriptors from both images
    #ratio_threshold is 0.75 is the default strictness for Lowe's ratio test
    #lower number means stricter matching, fewer but more reliable matches

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    #this creates the bruteforce matcher
    #cv2.NORM_hamming tells it to use Hamming distance to comparte the decriptors. This is the correct metric for ORB binary descriptors. Hamming distance calculates how many bits differ between two descriptors.
    #crossCheck = False because we are using knnMatch with ratio test instead of simple match

    raw_matches = matcher.knnMatch(desc1, desc2, k=2)
    #this means for every descriptor in desc1, find the 2 closest descriptors in desc2
    #k=2 means return the 2 best matches per descriptor
    #m=best match, n= 2nd best match

    good_matches = []

    for m, n in raw_matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)
    # this is the Lowe's ratio test. Loop through every pair of (best, second best) matches and if the best match is significantly closer than the second best, keep it. Otherwise, discard it as ambiguos. 
    #as an analogy: if 65 people say turn left and 15 say random things, you turn left. But if 40 say left and 35 say right, you don't trust either. 

    print(f"raw matches: {len(raw_matches)}")
    print(f"good matches after ratio test: {len(good_matches)}")
    return good_matches


def visualize_matches(img1_color, img2_color, kp1, kp2, good_matches, max_display=60):
    match_img = cv2.drawMatches( #creates a new wide image by placing both photos side by side and drawing lines connecting each matched keypoint pair
        img1_color, kp1,
        img2_color, kp2,
        good_matches[:max_display], None, #good_matches[:max_display] means only draw the first 60 matches so lines don't overlap into a mess
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS #hides keypoints that don't have a match
    )
    match_rgb = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(16, 6))
    plt.imshow(match_rgb)
    plt.title(f"Step 3: Keypoint Matching — {len(good_matches)} good matches", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("output/step3_matches.jpg", dpi=150, bbox_inches='tight')
    plt.show()
    print("saved: output/step3_matches.jpg")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, 'src')
    from step1_load import load_images
    from step2_detect import detect_keypoints
    img1c, img2c, img1g, img2g = load_images("images/img1.jpg", "images/img2.jpg")
    kp1, desc1, kp2, desc2 = detect_keypoints(img1g, img2g)
    good = match_keypoints(desc1, desc2)
    visualize_matches(img1c, img2c, kp1, kp2, good)