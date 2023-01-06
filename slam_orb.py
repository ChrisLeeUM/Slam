import cv2
import numpy as np

# extract key-points
def extract_points(frame):
    orb = cv2.ORB_create()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detection corners
    pts = cv2.goodFeaturesToTrack(image, 3000, qualityLevel=0.01, minDistance=3)
    # extract features
    kps = [cv2.KeyPoint(x=pt[0][0], y=pt[0][1], size=20) for pt in pts]
    kps, des = orb.compute(image, kps)

    kps = np.array([(kp.pt[0], kp.pt[1]) for kp in kps])
    return kps, des

if __name__ == "__main__":
    cap = cv2.VideoCapture("road.mp4")
    while cap.isOpened() :
        ret, image = cap.read()

        if ret:
            kps, des = extract_points(image)
            for i in range(500):
                u1, v1 = int(kps[i][0]), int(kps[i][1])
                cv2.circle(image, (u1, v1), color=(0,0,255), radius=3)
            # print(kps[0][0], kps[0][1])
            cv2.imshow("slam", image)

        # print(len(kps), kps)
        # print(len(des), des)
        if cv2.waitKey(30) & 0xFF == ord('q'): break
