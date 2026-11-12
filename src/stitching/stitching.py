import cv2
import numpy as np

# This code is inspired by the example on this page:
# https://www.geeksforgeeks.org/python/opencv-panorama-stitching/
# accessed on the 10-11-2026
def stitching_built_in(frames):
    stitchy = cv2.Stitcher.create(cv2.STITCHER_PANORAMA)
    # Parameters given by gemini
    stitchy.setPanoConfidenceThresh(0.5)
    stitchy.setRegistrationResol(0.25)
    stitchy.setSeamEstimationResol(0.1)
    stitchy.setCompositingResol(-1)
    stitchy.setWaveCorrection(True)

    (dummy, panorama) = stitchy.stitch(frames)
    if dummy == cv2.STITCHER_OK:
        frames_in = stitchy.component()
        print(frames_in.flatten())
    else:
        print(dummy)
        print("Problem: sticher failed")
    
    return panorama


# https://blog.thoughtsre.com/p/codewalkthrough-image-stitching-with
# https://www.opencvhelp.org/tutorials/advanced/image-stitching/
# accessed on the 12-11-2026
def stitch(frames, nfeatures=4000, k=2, Lowe_threshold=0.75, Ransac_threshold=5):
    # Initialization
    orb = cv2.ORB_create(nfeatures=nfeatures)

    # Parameter given by gemini
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
    # Alternative
    #bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    panorama = frames[0]

    for i, frame in enumerate(frames[1:], start=1):  
        # Features detection  
        
        panorama_res = orb.detectAndCompute(panorama, None)
        frame_res = orb.detectAndCompute(frame, None)

        # Features matching
        matches = [_ for _ in flann.knnMatch(panorama_res[1], frame_res[1], k=k) if len(_) == 2]
        # Alternative
        # matches = [_ for _ in bf.knnMatch(panorama_res[1], frame_res[1], k=k) if len(_) == 2]

        good_matches = []

        for m, n in matches:
            if m.distance < Lowe_threshold * n.distance:
                good_matches.append(m)

        good_matches = sorted(good_matches, key=lambda t: t.distance)

        if len(good_matches) < 4:
            print("Not enough good matches")
            continue

        # Homography
        src_pts = np.float32([frame_res[0][m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([panorama_res[0][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, Ransac_threshold)

        # 1st check if the skew is not too big
        if H is not None:
            det = np.linalg.det(H[:2, :2])
            if det < 0.5 or det > 2.0 or abs(H[2, 0]) > 0.002 or abs(H[2, 1]) > 0.002:
                print("problem with perspective")
                continue
        else:
            print("problem with H")
            continue

        # Warping
        h1, w1 = panorama.shape[:2]
        h2, w2 = frame.shape[:2]

        panorama_corners = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape((-1, 1, 2))
        frame_corners = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape((-1, 1, 2))

        warped_frame_corners = cv2.perspectiveTransform(frame_corners, H)

        all_corners = np.concatenate((panorama_corners, warped_frame_corners), axis=0)

        [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

        t_x = -xmin if xmin < 0 else 0
        t_y = -ymin if ymin < 0 else 0

        w_prime = xmax + t_x
        h_prime = ymax + t_y

        Ht =np.array([[1, 0, t_x],
                      [0, 1, t_y],
                      [0, 0, 1]])

        frame_warped = cv2.warpPerspective(frame, Ht @ H, (w_prime, h_prime))

        # Check to not have 9.8 Gb in memory (yes it happened)
        max_allowed_w = w1 + w2
        max_allowed_h = h1 + h2  
        if w_prime <= 0 or h_prime <= 0 or w_prime > max_allowed_w or h_prime > max_allowed_h:
            print("Extreme warp dimensions")
            continue

        pano_canvas = np.zeros_like(frame_warped)
        pano_canvas[t_y:h1 + t_y, t_x:w1 + t_x] = panorama

        # Blending
        mask = (pano_canvas > 0).astype(np.uint8)
        blended_img = pano_canvas * mask + frame_warped * (1 - mask)
        panorama = blended_img.astype(np.uint8)

    return panorama
