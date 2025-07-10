import cv2
import mediapipe as mp
import edge_tts
import asyncio
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

hand_state = 0

text=''
text2 = ''

r=0
p=0

hi_deviation=0.05

cap = cv2.VideoCapture(0)

mp1_drawing = mp.solutions.drawing_utils
mp1_pose = mp.solutions.pose
pose = mp1_pose.Pose(model_complexity=0,min_detection_confidence=0.5, min_tracking_confidence=0.5)

x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
y = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

x1 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
y1 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]

def speak(speech, voice="en-US-guyNeural", output_file="output.mp3"):
    async def create_audio():
        tts = edge_tts.Communicate(speech, voice)
        await tts.save(output_file)
    asyncio.run(create_audio())
    os.system(f"mpg123 {output_file}")
    os.remove(output_file)

def bye_true(x, y):
    if (y[8]>y[5] and y[12]>y[9] and y[16]>y[13] and y[20]>y[17]) and x[5]>x[4]:
        return 1

def FU_true(x, y):
    if y[8]>y[5] and y[12]<y[9] and y[16]>y[13] and y[20]>y[17]:
        return 1

def reset_true(x, y):
    if (y[8]<y[5] and y[12]<y[9] and y[16]<y[13] and y[20]<y[17]) and (x[5] >x[4]):
        return 1

def my_true(x, y, x1, y1):
    if (x[8]>x[5] and x[12]>x[9] and x[16]>x[13] and x[20]>x[17]) and (x1[19]>x1[12]) and (y1[19]>y1[12]):
        return 1
def hi_true(p):
    if p == 1:
        return 1

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
    
        if not success:
            print("Ignoring empty camera frame.")
            continue

        _, frame = cap.read()
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(frame_rgb)
            mp1_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp1_pose.POSE_CONNECTIONS)
        except:
            break

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks and pose_results.pose_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,

                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                zero_landmark = hand_landmarks.landmark[0]
                x[0] = zero_landmark.x
                y[0] = zero_landmark.y

                one_landmark = hand_landmarks.landmark[1]
                x[1] = one_landmark.x
                y[1] = one_landmark.y

                two_landmark = hand_landmarks.landmark[2]
                x[2] = two_landmark.x
                y[2] = two_landmark.y

                three_landmark = hand_landmarks.landmark[3]
                x[3] = three_landmark.x
                y[3] = three_landmark.y

                four_landmark = hand_landmarks.landmark[4]
                x[4] = four_landmark.x
                y[4] = four_landmark.y

                five_landmark = hand_landmarks.landmark[5]
                x[5] = five_landmark.x
                y[5] = five_landmark.y

                six_landmark = hand_landmarks.landmark[6]
                x[6] = six_landmark.x
                y[6] = six_landmark.y

                seven_landmark = hand_landmarks.landmark[7]
                x[7] = seven_landmark.x
                y[7] = seven_landmark.y

                eight_landmark = hand_landmarks.landmark[8]
                x[8] = eight_landmark.x
                y[8] = eight_landmark.y

                nine_landmark = hand_landmarks.landmark[9]
                x[9] = nine_landmark.x
                y[9] = nine_landmark.y

                ten_landmark = hand_landmarks.landmark[10]
                x[10] = ten_landmark.x
                y[10] = ten_landmark.y

                eleven_landmark = hand_landmarks.landmark[11]
                x[11] = eleven_landmark.x
                y[11] = eleven_landmark.y

                twelve_landmark = hand_landmarks.landmark[12]
                x[12] = twelve_landmark.x
                y[12] = twelve_landmark.y

                thirteen_landmark = hand_landmarks.landmark[13]
                x[13] = thirteen_landmark.x
                y[13] = thirteen_landmark.y

                fourteen_landmark = hand_landmarks.landmark[14]
                x[14] = fourteen_landmark.x
                y[14] = fourteen_landmark.y

                fifteen_landmark = hand_landmarks.landmark[15]
                x[15] = fifteen_landmark.x
                y[15] = fifteen_landmark.y

                sixteen_landmark = hand_landmarks.landmark[16]
                x[16] = sixteen_landmark.x
                y[16] = sixteen_landmark.y

                seventeen_landmark = hand_landmarks.landmark[17]
                x[17] = seventeen_landmark.x
                y[17] = seventeen_landmark.y

                eighteen_landmark = hand_landmarks.landmark[18]
                x[18] = eighteen_landmark.x
                y[18] = eighteen_landmark.y

                nineteen_landmark = hand_landmarks.landmark[19]
                x[19] = nineteen_landmark.x
                y[19] = nineteen_landmark.y

                twenty_landmark = hand_landmarks.landmark[20]
                x[20] = twenty_landmark.x
                y[20] = twenty_landmark.y

                arm_right_landmark = pose_results.pose_landmarks.landmark[16]
                x1[19] = arm_right_landmark.x
                y1[19] = arm_right_landmark.y

                chest_right_landmark = pose_results.pose_landmarks.landmark[12]
                x1[12] = chest_right_landmark.x
                y1[12] = chest_right_landmark.y    

                waist_right_landmark = pose_results.pose_landmarks.landmark[24]
                x1[24] = waist_right_landmark.x
                y1[24] = waist_right_landmark.y    

                if bye_true(x, y):
                    text = 'bye'

                if FU_true(x, y):
                    text='fuck you'

                if reset_true(x, y):
                    p=1
                    xs=x[9]
                    text= ''
                    text2=''
                if hi_true(p):
                    print(r)
                    r+=1
                    if (r < 20)and (xs-x[9] > hi_deviation or xs-x[9] < -hi_deviation) :
                    
                        text='hi'
                        p=0
                    elif r >= 20:
                        r=0
                        p=0
                if my_true(x, y, x1, y1):
                    text2='my'
                    text='my'

        font = cv2.FONT_HERSHEY_SIMPLEX 
   
        flipped_image = cv2.flip(image, 1)   
        flipped_image2 = cv2.flip(frame, 1) 

        cv2.putText(flipped_image, text, (50, 100), font, 1.5, (0, 0, 0), 3)
        cv2.putText(flipped_image2, text2, (50, 100), font, 1.5, (0, 0, 0), 3)

        cv2.imshow('MediaPipe Hands',flipped_image)
        cv2.imshow('MEdiapipe body',flipped_image2)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
cap.release()      
       
    