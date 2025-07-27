import cv2
import mediapipe as mp
import time
import os
hand_state = 0

text=''
text2 = ''
th=0
xh=0
yh=0
thh=0
r=0
p=0
wap=0
hi_dev=0.1

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose




cap = cv2.VideoCapture(0)


pose = mp_pose.Pose(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)


text = ''
text2 = ''


x1 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
y1 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
def a_true(x_r, y_r):
    if (y_r[7]>y_r[6] and y_r[11]>y_r[10] and y_r[15]>y_r[14] and y_r[19]>y_r[18]) and (y_r[8]>y_r[7] and y_r[12]>y_r[11] and y_r[16]>y_r[15] and y_r[20]>y_r[19])and y_r[5]>y_r[4] and y_r[3]>y_r[4]:
        return 1

def FU_true(x_r, y_r):
    if y_r[8]>y_r[5] and y_r[12]<y_r[9] and y_r[16]>y_r[13] and y_r[20]>y_r[17]:
        return 1

def b_true(x_r, y_r):
    if (y_r[8]<y_r[5] and y_r[12]<y_r[9] and y_r[16]<y_r[13] and y_r[20]<y_r[17]) and (x_r[5] >x_r[4]):
        return 1
def reset_true(x_l, y_l):
    if (y_l[8]<y_l[5] and y_l[12]<y_l[9] and y_l[16]<y_l[13] and y_l[20]<y_l[17]) and (x_l[5] <x_l[4]):
        return 1
def my_true(x_r, y_r, x1, y1):
    if (x_r[8]>x_r[5] and x_r[12]>x_r[9] and x_r[16]>x_r[13] and x_r[20]>x_r[17]) and (x1[19]>x1[12]) and (y1[19]>y1[12]):
        return 1
def hi_true(p):
    if p == 1:
        return 1
def name_true(y_l,x1, y1):
    if (y_l[5]-y_l[6])<0.02 and ((x1[19]>x1[12]) and (y1[19]>y1[12])):
        return 1
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(frame_rgb)

        
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        
        x_l= [0] * 21
        y_l = [0] * 21
        x_r = [0] * 21
        y_r = [0] * 21

        
        if results.multi_hand_landmarks and pose_results.pose_landmarks:
            for idx , handedness in enumerate(results.multi_handedness):
                label = handedness.classification[0].label  
                hand_landmarks = results.multi_hand_landmarks[idx]

                
                if label == 'Left':
                    zero_r_landmark = hand_landmarks.landmark[0]
                    x_r[0] = zero_r_landmark.x
                    y_r[0] = zero_r_landmark.y

                    one_r_landmark = hand_landmarks.landmark[1]
                    x_r[1] = one_r_landmark.x
                    y_r[1] = one_r_landmark.y

                    two_r_landmark = hand_landmarks.landmark[2]
                    x_r[2] = two_r_landmark.x
                    y_r[2] = two_r_landmark.y

                    three_r_landmark = hand_landmarks.landmark[3]
                    x_r[3] = three_r_landmark.x
                    y_r[3] = three_r_landmark.y

                    four_r_landmark = hand_landmarks.landmark[4]
                    x_r[4] = four_r_landmark.x
                    y_r[4] = four_r_landmark.y

                    five_r_landmark = hand_landmarks.landmark[5]
                    x_r[5] = five_r_landmark.x
                    y_r[5] = five_r_landmark.y

                    six_r_landmark = hand_landmarks.landmark[6]
                    x_r[6] = six_r_landmark.x
                    y_r[6] = six_r_landmark.y

                    seven_r_landmark = hand_landmarks.landmark[7]
                    x_r[7] = seven_r_landmark.x
                    y_r[7] = seven_r_landmark.y

                    eight_r_landmark = hand_landmarks.landmark[8]
                    x_r[8] = eight_r_landmark.x
                    y_r[8] = eight_r_landmark.y

                    nine_r_landmark = hand_landmarks.landmark[9]
                    x_r[9] = nine_r_landmark.x
                    y_r[9] = nine_r_landmark.y

                    ten_r_landmark = hand_landmarks.landmark[10]
                    x_r[10] = ten_r_landmark.x
                    y_r[10] = ten_r_landmark.y

                    eleven_r_landmark = hand_landmarks.landmark[11]
                    x_r[11] = eleven_r_landmark.x
                    y_r[11] = eleven_r_landmark.y

                    twelve_r_landmark = hand_landmarks.landmark[12]
                    x_r[12] = twelve_r_landmark.x
                    y_r[12] = twelve_r_landmark.y

                    thirteen_r_landmark = hand_landmarks.landmark[13]
                    x_r[13] = thirteen_r_landmark.x
                    y_r[13] = thirteen_r_landmark.y

                    fourteen_r_landmark = hand_landmarks.landmark[14]
                    x_r[14] = fourteen_r_landmark.x
                    y_r[14] = fourteen_r_landmark.y

                    fifteen_r_landmark = hand_landmarks.landmark[15]
                    x_r[15] = fifteen_r_landmark.x
                    y_r[15] = fifteen_r_landmark.y

                    sixteen_r_landmark = hand_landmarks.landmark[16]
                    x_r[16] = sixteen_r_landmark.x
                    y_r[16] = sixteen_r_landmark.y

                    seventeen_r_landmark = hand_landmarks.landmark[17]
                    x_r[17] = seventeen_r_landmark.x
                    y_r[17] = seventeen_r_landmark.y

                    eighteen_r_landmark = hand_landmarks.landmark[18]
                    x_r[18] = eighteen_r_landmark.x
                    y_r[18] = eighteen_r_landmark.y

                    nineteen_r_landmark = hand_landmarks.landmark[19]
                    x_r[19] = nineteen_r_landmark.x
                    y_r[19] = nineteen_r_landmark.y

                    twenty_r_landmark = hand_landmarks.landmark[20]
                    x_r[20] = twenty_r_landmark.x
                    y_r[20] = twenty_r_landmark.y


                elif label == 'Right':
                    zero_l_landmark = hand_landmarks.landmark[0]
                    x_l[0] = zero_l_landmark.x
                    y_l[0] = zero_l_landmark.y

                    one_l_landmark = hand_landmarks.landmark[1]
                    x_l[1] = one_l_landmark.x
                    y_l[1] = one_l_landmark.y

                    two_l_landmark = hand_landmarks.landmark[2]
                    x_l[2] = two_l_landmark.x
                    y_l[2] = two_l_landmark.y

                    three_l_landmark = hand_landmarks.landmark[3]
                    x_l[3] = three_l_landmark.x
                    y_l[3] = three_l_landmark.y

                    four_l_landmark = hand_landmarks.landmark[4]
                    x_l[4] = four_l_landmark.x
                    y_l[4] = four_l_landmark.y

                    five_l_landmark = hand_landmarks.landmark[5]
                    x_l[5] = five_l_landmark.x
                    y_l[5] = five_l_landmark.y

                    six_l_landmark = hand_landmarks.landmark[6]
                    x_l[6] = six_l_landmark.x
                    y_l[6] = six_l_landmark.y

                    seven_l_landmark = hand_landmarks.landmark[7]
                    x_l[7] = seven_l_landmark.x
                    y_l[7] = seven_l_landmark.y

                    eight_l_landmark = hand_landmarks.landmark[8]
                    x_l[8] = eight_l_landmark.x
                    y_l[8] = eight_l_landmark.y

                    nine_l_landmark = hand_landmarks.landmark[9]
                    x_l[9] = nine_l_landmark.x
                    y_l[9] = nine_l_landmark.y

                    ten_l_landmark = hand_landmarks.landmark[10]
                    x_l[10] = ten_l_landmark.x
                    y_l[10] = ten_l_landmark.y

                    eleven_l_landmark = hand_landmarks.landmark[11]
                    x_l[11] = eleven_l_landmark.x
                    y_l[11] = eleven_l_landmark.y

                    twelve_l_landmark = hand_landmarks.landmark[12]
                    x_l[12] = twelve_l_landmark.x
                    y_l[12] = twelve_l_landmark.y

                    thirteen_l_landmark = hand_landmarks.landmark[13]
                    x_l[13] = thirteen_l_landmark.x
                    y_l[13] = thirteen_l_landmark.y

                    fourteen_l_landmark = hand_landmarks.landmark[14]
                    x_l[14] = fourteen_l_landmark.x
                    y_l[14] = fourteen_l_landmark.y

                    fifteen_l_landmark = hand_landmarks.landmark[15]
                    x_l[15] = fifteen_l_landmark.x
                    y_l[15] = fifteen_l_landmark.y

                    sixteen_l_landmark = hand_landmarks.landmark[16]
                    x_l[16] = sixteen_l_landmark.x
                    y_l[16] = sixteen_l_landmark.y

                    seventeen_l_landmark = hand_landmarks.landmark[17]
                    x_l[17] = seventeen_l_landmark.x
                    y_l[17] = seventeen_l_landmark.y

                    eighteen_l_landmark = hand_landmarks.landmark[18]
                    x_l[18] = eighteen_l_landmark.x
                    y_l[18] = eighteen_l_landmark.y

                    nineteen_l_landmark = hand_landmarks.landmark[19]
                    x_l[19] = nineteen_l_landmark.x
                    y_l[19] = nineteen_l_landmark.y

                    twenty_l_landmark = hand_landmarks.landmark[20]
                    x_l[20] = twenty_l_landmark.x
                    y_l[20] = twenty_l_landmark.y


                
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            
            arm_right_landmark = pose_results.pose_landmarks.landmark[16]
            x1[19] = arm_right_landmark.x
            y1[19] = arm_right_landmark.y

            chest_right_landmark = pose_results.pose_landmarks.landmark[12]
            x1[12] = chest_right_landmark.x
            y1[12] = chest_right_landmark.y    

            waist_right_landmark = pose_results.pose_landmarks.landmark[24]
            x1[24] = waist_right_landmark.x
            y1[24] = waist_right_landmark.y

            right_ear = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR]
            


            if a_true(x_r, y_r):
                text = 'A'

            if FU_true(x_r, y_r):
                text='fuck you'

            if b_true(x_r, y_r):
                text= 'B'
            if reset_true(x_l, y_l):
                text= ''
                text2=''

            if my_true(x_r, y_r, x1, y1):
                text2='my'
                text='my'
            if name_true(y_l, x1, y1):
                text2='name'
                text='name'
            #print("Right Ear X:", right_ear.x, "Y:", right_ear.y)
            #print("Right Hand:", x_r[8], y_r[8])
            #print((right_ear.x-x_r[8])) 
            if (right_ear.x-x_r[8])> 0.005 and (right_ear.x-x_r[8])< 0.03 and hi_true:
                if thh==1:
                    thh=0
                th=1
                xh=x_r[8]
                yh=y_r[8]
                wap+=1
                print(wap)
               
                time.sleep(0.05)
            
            if th ==1 and xh!=0 and yh!=0 and thh==0:
                wap+=1
                print(wap)

                time.sleep(0.05)
                if wap>8 and wap<50:
           
                    if (xh-x_r[8])>-hi_dev :
                        text = 'hi'
                        thh=1
                        wap=0
                        
                if wap>50:
                    wap=0
                    thh=1                    


        font = cv2.FONT_HERSHEY_SIMPLEX
        flipped_image = cv2.flip(image, 1)
        flipped_frame = cv2.flip(frame, 1)

        cv2.putText(flipped_image, text, (50, 100), font, 1.5, (0, 0, 0), 3)
        cv2.putText(flipped_frame, text2, (50, 100), font, 1.5, (0, 0, 0), 3)

        cv2.imshow('MediaPipe Hands', flipped_image)
        #cv2.imshow('MediaPipe Body', flipped_frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()
