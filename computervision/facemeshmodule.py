import cv2
import mediapipe as mp
import time

class FaceMeshDetector:
    def __init__(self, staticImage=False, maxFaces=2, redefineLms=False, minDetectionCon=0.5, minTrackingCon=0.5):
        self.staticImage = staticImage
        self.maxFaces = maxFaces
        self.redefineLms = redefineLms
        self.minDetectionCon = minDetectionCon
        self.minTrackingCon = minTrackingCon

        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpecsBlue = self.mpDraw.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1)  # Blue color
        self.drawSpecsRed = self.mpDraw.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=2)  # Red color
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticImage, self.maxFaces, self.redefineLms,
                                                 self.minDetectionCon, self.minTrackingCon)

    def findFaceMesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)

        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                               self.drawSpecsBlue, self.drawSpecsRed)

                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y])

                faces.append(face)

        return img, faces

def main():
    pTime = 0
    cap = cv2.VideoCapture(0)

    detector = FaceMeshDetector()

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (700, 420))
        img, faces = detector.findFaceMesh(img)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Video", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
