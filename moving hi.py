import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hand_state = 0
text=''
r=0
p=0
t=10
hi_deviation= 0.3
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            
            continue

        
        
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,

                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                zero_landmark = hand_landmarks.landmark[0]
                x0 = zero_landmark.x
                y0 = zero_landmark.y

                one_landmark = hand_landmarks.landmark[1]
                x1 = one_landmark.x
                y1 = one_landmark.y

                two_landmark = hand_landmarks.landmark[2]
                x2 = two_landmark.x
                y2 = two_landmark.y

                three_landmark = hand_landmarks.landmark[3]
                x3 = three_landmark.x
                y3 = three_landmark.y

                four_landmark = hand_landmarks.landmark[4]
                x4 = four_landmark.x
                y4 = four_landmark.y

                five_landmark = hand_landmarks.landmark[5]
                x5 = five_landmark.x
                y5 = five_landmark.y

                six_landmark = hand_landmarks.landmark[6]
                x6 = six_landmark.x
                y6 = six_landmark.y

                seven_landmark = hand_landmarks.landmark[7]
                x7 = seven_landmark.x
                y7 = seven_landmark.y

                eight_landmark = hand_landmarks.landmark[8]
                x8 = eight_landmark.x
                y8 = eight_landmark.y

                nine_landmark = hand_landmarks.landmark[9]
                x9 = nine_landmark.x
                y9 = nine_landmark.y

                ten_landmark = hand_landmarks.landmark[10]
                x10 = ten_landmark.x
                y10 = ten_landmark.y

                eleven_landmark = hand_landmarks.landmark[11]
                x11 = eleven_landmark.x
                y11 = eleven_landmark.y

                twelve_landmark = hand_landmarks.landmark[12]
                x12 = twelve_landmark.x
                y12 = twelve_landmark.y

                thirteen_landmark = hand_landmarks.landmark[13]
                x13 = thirteen_landmark.x
                y13 = thirteen_landmark.y

                fourteen_landmark = hand_landmarks.landmark[14]
                x14 = fourteen_landmark.x
                y14 = fourteen_landmark.y

                fifteen_landmark = hand_landmarks.landmark[15]
                x15 = fifteen_landmark.x
                y15 = fifteen_landmark.y

                sixteen_landmark = hand_landmarks.landmark[16]
                x16 = sixteen_landmark.x
                y16 = sixteen_landmark.y

                seventeen_landmark = hand_landmarks.landmark[17]
                x17 = seventeen_landmark.x
                y17 = seventeen_landmark.y

                eighteen_landmark = hand_landmarks.landmark[18]
                x18 = eighteen_landmark.x
                y18 = eighteen_landmark.y

                nineteen_landmark = hand_landmarks.landmark[19]
                x19 = nineteen_landmark.x
                y19 = nineteen_landmark.y

                twenty_landmark = hand_landmarks.landmark[20]
                x20 = twenty_landmark.x
                y20 = twenty_landmark.y

                if y8>y5 and y12>y9 and y16>y13 and y20>y17:
                    print('close')
                    text='bye'
                
                    
                if y8<y5 and y12<y9 and y16<y13 and y20<y17:
                    print('open')
                    #hi
                    if x5 >x4 :
                        p=1
                        xs=x9
                        text= ''
                if p ==1:
                    print(r)
                    r+=1
                    if (r < t*10)and (xs-x9 > hi_deviation or xs-x9 < -hi_deviation):
                    
                        text='hi'
                        p=0
                    if r >= t*10:
                        r=0
                        p=0


                    


        font = cv2.FONT_HERSHEY_SIMPLEX
            
        flipped_image = cv2.flip(image, 1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
        cv2.putText(flipped_image, text, (50, 100), font, 1.5, (0, 0, 0), 3)
        
        cv2.imshow('MediaPipe Hands',flipped_image )

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
cap.release()   