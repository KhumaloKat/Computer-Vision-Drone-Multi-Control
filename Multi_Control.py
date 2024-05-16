from djitellopy import tello
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.PoseModule import PoseDetector
import cvzone
import time

cap = cv2.VideoCapture(0)
detectorHand = HandDetector(maxHands=1, detectionCon=0.9)
detectorFace = FaceDetector()
detectorPose = PoseDetector()
h1, w1, _ = 480, 640, True

colourG = (0, 0, 255)
gesture = ''

###-Gesture Control-###
colourG_CL = (0, 0, 255)
gesture_CL = ''
gesture_C = False

###-Face Following-###
following = False

xPID = cvzone.PID([0.12, 0, 0.1], w1//2)
yPID = cvzone.PID([0.17, 0, 0.1], h1//2, axis=1)
# zPID = cvzone.PID([0.003, 0, 0.003], 12000, limit=[-20, 15])

###-Hand motion-###
motion = False

mxVal = 0
myVal = 0
mzVal = 0

mxPID = cvzone.PID([0.23, 0, 0.1], w1//2)
myPID = cvzone.PID([0.28, 0, 0.1], h1//2, axis=1)
mzPID = cvzone.PID([0.005, 0, 0.003], 12000, limit=[-20, 15])

myPlotx = cvzone.LivePlot(yLimit=[-100, 100], char='x')
myPloty = cvzone.LivePlot(yLimit=[-100, 100], char='Y')
myPlotz = cvzone.LivePlot(yLimit=[-100, 100], char='Z')

#Tello Enabling

# me = tello.Tello()
# me.connect()
# print(me.get_battery())
# me.streamoff()
# me.streamon()
# me.takeoff()
# me.move_up(90)

snapTime = 0


while True:
    _, img = cap.read()
    # img = me.get_frame_read().frame
    img = cv2.resize(img, (640, 480))
    img = detectorHand.findHands(img)
    lmList, bboxInfo = detectorHand.findPosition(img)
    img, bboxs = detectorFace.findFaces(img, draw=False)

    img = detectorPose.findPose(img, draw=False)
    lmLists, bboxInfos = detectorPose.findPosition(img, draw=False)

    if bboxs:
        cx, cy = bboxs[0]['center']
        x, y, w, h = bboxs[0]["bbox"]   # bounding box of the detected face
        bboxRegion = x - 175 - 25, y - 75, 175, h + 75
        bboxRegion2 = x + 175, y - 75, 175, h + 75

        area = w * h

        xVal = int(xPID.update(cx))
        yVal = int(yPID.update(cy))
        # zVal = int(zPID.update(area))

        if lmList and detectorHand.handType() == "Right":

            cvzone.cornerRect(img, bboxRegion, rt=0, t=10, colorC=(0, 0, 255))
            handCenter = bboxInfo[0]["center"]

            inside = bboxRegion[0] < handCenter[0] < bboxRegion[0] + bboxRegion[2] and \
                     bboxRegion[1] < handCenter[1] < bboxRegion[1] + bboxRegion[3]

            if inside:
                cvzone.cornerRect(img, bboxRegion, rt=0, t=10, colorC=(0, 255, 0))

                fingers = detectorHand.fingersUp()

                if fingers == [1, 1, 1, 1, 1]:
                    following = False
                    gesture_C = False
                    motion = False
                    gesture = "Initial State"
                    colourG = (0, 0, 255)

                elif fingers == [1, 0, 0, 0, 0]:
                    gesture_C = True
                    following = False
                    motion = False
                    gesture = "Gesture Control: ON"
                    colourG = (225, 105, 65)

                elif fingers == [0, 0, 1, 1, 1]:
                    gesture_C = False
                    following = False
                    motion = False
                    gesture = "Image Capture"
                    colourG = (225, 105, 65)
                    snapTime = time.time()


                elif fingers == [0, 0, 1, 1, 1]:
                    gesture_C = False
                    following = False
                    motion = False
                    gesture = "View Capture"
                    # me.move_back(80)
                    # me.move_up(40)
                    # snapTime = time.time()
                    # me.rotate_clockwise(90)
                    # snapTime = time.time()
                    # me.rotate_clockwise(90)
                    # snapTime = time.time()
                    # me.rotate_clockwise(90)
                    # snapTime = time.time()
                    # me.rotate_clockwise(90)
                    # me.move_down(40)
                    # me.move_forward(80)
                    colourG = (225, 105, 65)


                elif fingers == [0, 0, 0, 0, 0]:
                    gesture_C = False
                    following = False
                    motion = True
                    gesture = "Motion : ON"
                    colourG = (225, 105, 65)

                elif fingers == [1, 1, 0, 0, 1]:
                    gesture = "  Face Following: ON"
                    colourG = (225, 105, 65)
                    motion = False
                    gesture_C = False
                    following = True


        if snapTime > 0:
            totaltime = time.time() - snapTime

            if totaltime < 1.9:
                cv2.putText(img, " ", (120, 260), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

            if totaltime > 2:
                snapTime = 0
                cv2.imwrite(f'Images/{time.time()}.jpg', img)
                cv2.putText(img, "Saved", (120, 260), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    if gesture_C:

        cvzone.cornerRect(img, bboxRegion2, rt=0, t=10, colorC=(0, 0, 255))

        if lmList and detectorHand.handType() == "Left":

            handCenter2 = bboxInfo[0]["center"]

            inside = bboxRegion2[0] < handCenter2[0] < bboxRegion2[0] + bboxRegion2[2] and \
                         bboxRegion2[1] < handCenter2[1] < bboxRegion2[1] + bboxRegion2[3]

            if inside:

                cvzone.cornerRect(img, bboxRegion2, rt=0, t=10, colorC=(0, 255, 0))

                fingers = detectorHand.fingersUp()

                if fingers == [1, 1, 1, 1, 1]:
                    following = False
                    gesture_CL = "Stop"

                elif fingers == [0, 1, 0, 0, 0]:
                    following = False
                    gesture_CL = "UP"
                    #me.move_up(40)

                elif fingers == [0, 1, 1, 0, 0]:
                    following = False
                    gesture_CL = "Down"
                    #me.move_down(40)

                elif fingers == [0, 0, 0, 0, 1]:
                    following = False
                    gesture_CL = "Left"
                    #me.move_left(50)

                elif fingers == [1, 0, 0, 0, 0]:
                    following = False
                    gesture_CL = "Right"
                    #me.move_right(50)

                elif fingers == [1, 1, 0, 0, 1]:
                    gesture_CL = "  Foward"
                    following = True
                    #me.move_forward(50)

                elif fingers == [0, 1, 0, 0, 1]:
                    gesture_CL = "  Backward"
                    colourG_CL = (0, 255, 0)
                    following = True
                    #me.move_back(50)

                cv2.putText(img, f'{gesture_CL}',
                                (bboxRegion2[0] + 10, bboxRegion2[1] + bboxRegion2[3] + 50),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
    else:
        gesture_C = False

    if motion:

        if lmList and detectorHand.handType() == "Right":

            cx, cy = bboxInfo[0]["center"]
            x, y, w, h = bboxInfo[0]['bbox']
            area = w * h

            # Take all the values from the tello
            mxVal = int(mxPID.update(cx))
            myVal = int(myPID.update(cy))
            mzVal = int(mzPID.update(area))

            imgPlotx = myPlotx.update(mxVal)
            imgPloty = myPloty.update(myVal)
            imgPlotz = myPlotz.update(mzVal)
            img = mxPID.draw(img, [cx, cy])
            img = myPID.draw(img, [cx, cy])
            imageStacked = cvzone.stackImages([img, imgPlotx, imgPloty, imgPlotz], 2, 0.5)
            # cv2.putText(imageStacked, str(area), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            cv2.imshow("ImageStacked", imageStacked)

    cv2.putText(img, gesture, (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, colourG, 2)

    # #face_rc_to_Tello
    #
    # if following:
    #     me.send_rc_control(0, -zVal, -yVal, xVal)
    # else:
    #     me.send_rc_control(0, 0, 0, 0)
    #
    # #hand_rc_to_Tello
    #
    # if motion:
    #     me.send_rc_control(0, 0, -myVal, mxVal)
    # else:
    #     me.send_rc_control(0, 0, 0, 0)


    cv2.imshow("Image", img)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        #me.land()
        break
cv2.destroyAllWindows()
