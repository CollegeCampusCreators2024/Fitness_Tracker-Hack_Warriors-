import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import time
from PIL import Image
import requests
import pyttsx3
from twilio.rest import Client
from threading import Timer
#Use your respective API keys
TWILIO_ACCOUNT_SID = ''
TWILIO_AUTH_TOKEN = ''
TWILIO_PHONE_NUMBER = ''
NUTRITIONIX_APP_ID = ""
NUTRITIONIX_API_KEY = ''
NUTRITIONIX_API_URL = ""
SHEETY_API_URL = ""

def send_sms(exercise_name, reps, calories_burned, user_phone_number):
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    message = f"Exercise: {exercise_name}\nReps: {reps}\nCalories burned: {calories_burned:.2f}"

    try:
        client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=user_phone_number
        )
        st.success("SMS sent successfully!")
    except Exception as e:
        st.error(f"Failed to send SMS: {e}")

def get_calories_burned(exercise, weight, reps, height, age, gender):
    headers = {
        'x-app-id': NUTRITIONIX_APP_ID,
        'x-app-key': NUTRITIONIX_API_KEY,
        'Content-Type': 'application/json'
    }

    data = {
        "query": f"{reps} reps of {exercise}",
        "gender": gender,
        "weight_kg": weight,
        "height_cm": height,
        "age": age
    }

    response = requests.post(NUTRITIONIX_API_URL, headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        total_calories = result["exercises"][0]["nf_calories"]
        calories_per_rep = total_calories / reps
        return calories_per_rep
    else:
        st.error("Error fetching data from Nutritionix API.")
        return 0

def log_to_sheet(exercise, weight, reps, height, age, gender, calories_burned):
    headers = {'Content-Type': 'application/json'}
    data = {
        "sheet1": {
            "exercise": exercise,
            "weight": weight,
            "reps": reps,
            "height": height,
            "age": age,
            "gender": gender,
            "calories": calories_burned
        }
    }

    response = requests.post(SHEETY_API_URL, headers=headers, json=data)

    if response.status_code == 201:
        st.success("Data logged to Google Sheets successfully.")
    else:
        st.error("Failed to log data to Google Sheets.")

def inject_custom_css():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('https://images.pexels.com/photos/1229356/pexels-photo-1229356.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


class ExerciseCounter:
    def __init__(self, exercise_name, user_weight, user_height, user_age, gender, user_phone_number, frame_width=640, frame_height=480):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        self.count = 0
        self.calories_burned = 0
        self.exercise_name = exercise_name
        self.user_weight = user_weight
        self.user_height = user_height
        self.user_age = user_age
        self.gender = gender
        self.user_phone_number = user_phone_number
        self.is_active = False
        self.prev_angle = None
        self.prev_angle1 = None
        self.calories_per_rep = 0
        self.engine = pyttsx3.init()

    def calculate_angle(self, a, b, c):
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])

        radians = np.arccos(np.dot(b - a, c - b) / (np.linalg.norm(b - a) * np.linalg.norm(c - b)))
        angle = np.degrees(radians)
        return angle

    def process_frame(self, placeholder, counter_placeholder):
        with self.mp_pose.Pose(model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while self.cap.isOpened():
                ret, image = self.cap.read()
                if not ret:
                    break
                image = cv2.flip(image, 1)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                if results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                    self.update_count(results, counter_placeholder)

                img_display = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                placeholder.image(img_display, use_column_width=True)

                time.sleep(0.03)

    def update_count(self, results, counter_placeholder):
        left_shoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_elbow = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
        left_wrist = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_shoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_elbow = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
        right_wrist = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]

        angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        angle1 = self.calculate_angle(right_shoulder, right_elbow, right_wrist)

        if self.prev_angle is not None:
            delta = abs(angle - self.prev_angle)
            delta1 = abs(angle1 - self.prev_angle1)

            if not self.is_active and delta > 15 and delta1 > 15:
                self.is_active = True
            elif self.is_active and delta < 7 and delta1 < 7:
                self.is_active = False

        self.prev_angle = angle
        self.prev_angle1 = angle1

        if self.is_active and angle1 > 120 and angle > 120:
            self.count += 1

            if self.count == 1:
                self.calories_per_rep = get_calories_burned(self.exercise_name, self.user_weight, self.count, self.user_height, self.user_age, self.gender)

            self.calories_burned += self.calories_per_rep

            counter_placeholder.text(
                f"{self.exercise_name} Count: {self.count} | Calories Burned: {self.calories_burned:.2f}")

            if self.count % 15 == 0:
                self.engine.say("Set completed! Ready for new set")
                self.engine.runAndWait()
                log_to_sheet(self.exercise_name, self.user_weight, self.count, self.user_height, self.user_age,
                             self.gender, self.calories_burned)

    def stop(self):
        self.cap.release()
        st.warning("Exercise stopped.")
        # Send the SMS when exercise is stopped
        send_sms(self.exercise_name, self.count, self.calories_burned, self.user_phone_number)
        # Log the data to Google Sheets one last time
        log_to_sheet(self.exercise_name, self.user_weight, self.count, self.user_height, self.user_age,
                     self.gender, self.calories_burned)

def main():
    inject_custom_css()

    st.title("Fitness Tracker")
    placeholder = st.empty()
    counter_placeholder = st.empty()

    st.sidebar.header("User Details")
    exercise_name = st.sidebar.selectbox("Select Exercise", ["bicep curls", "lateral raises"])
    user_weight = st.sidebar.slider("Weight (kg)", 40, 150, 70)
    user_height = st.sidebar.slider("Height (cm)", 140, 200, 170)
    user_age = st.sidebar.slider("Age", 18, 65, 25)
    gender = st.sidebar.radio("Gender", ["male", "female"])
    user_phone_number = st.sidebar.text_input("Enter your phone number", max_chars=13)

    if st.sidebar.button("Start Exercise"):
        st.session_state.counter = ExerciseCounter(exercise_name, user_weight, user_height, user_age, gender, user_phone_number)
        st.session_state.process_thread = Timer(0, st.session_state.counter.process_frame, args=[placeholder, counter_placeholder])
        st.session_state.process_thread.start()

    if st.sidebar.button("Stop Exercise"):
        if 'counter' in st.session_state:
            st.session_state.counter.stop()
        else:
            st.warning("You need to start the exercise first.")

if __name__ == "__main__":
    main()
