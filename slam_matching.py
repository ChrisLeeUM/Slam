import cv2
import numpy as np
from matplotlib import pyplot as plt


# extract key-points
def extract_points(frame):
    orb = cv2.ORB_create()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detection corners
    pts = cv2.goodFeaturesToTrack(image, 3000, qualityLevel=0.01, minDistance=3)
    # extract features
    kps = [cv2.KeyPoint(x=pt[0][0], y=pt[0][1], size=20) for pt in pts]
    kps, des = orb.compute(image, kps)

    # kps = np.array([(kp.pt[0], kp.pt[1]) for kp in kps])
    return kps, des

class Frame():
    idx = 0
    frame, kps, des = None, None, None
    def __init__(self, frame):
        self.frame = frame
        self.kps, self.des = extract_points(frame)

if __name__ == "__main__":
    cap = cv2.VideoCapture("road.mp4")
    last_frame = None
    while cap.isOpened() :
        ret, image = cap.read()

        if ret:
            if last_frame == None:
                last_frame = Frame(image)
                continue
            current_frame = Frame(image)
            bfmatch = cv2.BFMatcher(cv2.NORM_HAMMING)
            res = bfmatch.knnMatch(current_frame.des, last_frame.des, k=2)
            good_match = []
            for m, n in res:
                if m.distance < 0.5 * n.distance:
                    good_match.append([m])
            img3 = cv2.drawMatchesKnn(last_frame.frame, last_frame.kps, current_frame.frame, current_frame.kps, good_match, None, flags=2)
            cv2.imshow("slam", img3)
            last_frame = current_frame
            # plt.imshow(img3),plt.show()
            # break

        if cv2.waitKey(30) & 0xFF == ord('q'): break
