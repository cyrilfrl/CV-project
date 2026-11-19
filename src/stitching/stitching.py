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

# The two following function drawn inspiration from the three following website.
# https://blog.thoughtsre.com/p/codewalkthrough-image-stitching-with
# https://www.opencvhelp.org/tutorials/advanced/image-stitching/
# accessed on the 12-11-2026
# https://www.geeksforgeeks.org/data-science/feature-matching-in-opencv/
# accessed on the 19-11-2026
def get_homeography(current, previous, nfeatures=4000, k=2, Lowe_threshold=0.75, Ransac_threshold=5.0):
    orb = cv2.ORB_create(nfeatures=nfeatures)

    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=dict(checks=50))

    # Feature detection
    kp_previous, des_previous = orb.detectAndCompute(previous, None)
    kp_current, des_current = orb.detectAndCompute(current, None)
   
    if des_previous is None or des_current is None or len(des_previous) < 4 or len(des_current) < 4:
        return np.eye(3, dtype=np.float32)

    # Matching
    matches = flann.knnMatch(des_previous, des_current, k=k)
    good_matches = []
    for match in matches:
        if len(match) == 2:
            m, n = match
            if m.distance < Lowe_threshold * n.distance:
                good_matches.append(m)
    
    if len(good_matches) < 4:
        return np.eye(3, dtype=np.float32)

    # Homography
    src_pts = np.float32([kp_current[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_previous[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=Ransac_threshold)
    if M is None:
        return np.eye(3, dtype=np.float32)

    H = np.eye(3, dtype=np.float32)
    H[:2, :] = M
    return H


def stitch(frames):
    ## Initialization
    h_img, w_img = frames[0].shape[:2]
    # Panorama size
    pano_w, pano_h = w_img * 4, int(h_img * 1.5)

    # Placement of the 1st frame on the left of the panorama
    T_offset = np.array([
        [1, 0, w_img // 2],
        [0, 1,  h_img // 4],
        [0, 0, 1]
    ], dtype=np.float32)

    # Set 1st frame on the panorama
    first_warped = cv2.warpPerspective(frames[0], T_offset, (pano_w, pano_h))
    seen_mask = cv2.warpPerspective(
        np.ones((h_img, w_img), dtype=np.uint8) * 255, T_offset, (pano_w, pano_h)
    )

    # Background
    clean_panorama = first_warped.copy()

    # 1st Homoegraphy matrix
    H_t_to_0 = np.eye(3, dtype=np.float32)

    inner_mask = cv2.erode(
        np.ones((h_img, w_img), dtype=np.uint8) * 255,
        np.ones((15, 15), np.uint8)
    )

    for idx, frame in enumerate(frames):
        print(f"Frame {idx + 1}/{len(frames)}")

        # Get the homeography matrix 
        if idx > 0:
            H_step = get_homeography(frame, frames[idx - 1])
            H_t_to_0 = H_t_to_0 @ H_step
        H_to_pano = T_offset @ H_t_to_0
        # Warp the current frame
        warped_frame = cv2.warpPerspective(frame, H_to_pano, (pano_w, pano_h))

        # Zone where we can blend the frame safely, flags=cv2.INTER_NEAREST remove some black lines
        valid_zone = cv2.warpPerspective(inner_mask, H_to_pano, (pano_w, pano_h), flags=cv2.INTER_NEAREST)
        
        clean_panorama[valid_zone > 0] = warped_frame[valid_zone > 0]
        
        # Update pixels seen in the mask
        seen_mask[valid_zone > 0] = 255

    # Crop the panorama to only keep the part with an actual image
    coords = cv2.findNonZero(seen_mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        clean_panorama = clean_panorama[y:y+h, x:x+w]

    return clean_panorama 