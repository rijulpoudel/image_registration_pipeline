import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def match_keypoints(desc1, desc2, ratio_threshold=0.75): 
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