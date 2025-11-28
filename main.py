import streamlit as st
import face_recognition
import numpy as np
import pandas as pd
import pickle
import cv2
from datetime import datetime
import os
import pyttsx3
import threading
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av

ENCODINGS_FILE = 'face_encodings.pkl'
ATTENDANCE_FILE = 'attendance.csv'

# Function to speak text
def speak(text):
    def run_speak():
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            st.error(f"Error in text-to-speech: {e}")

    thread = threading.Thread(target=run_speak, daemon=True)
    thread.start()


# Helper to safely rerun Streamlit (some Streamlit builds remove experimental_rerun)
def safe_rerun():
    try:
        st.experimental_rerun()
    except Exception:
        try:
            st.experimental_set_query_params(_ts=int(time.time()))
        except Exception:
            # last resort: toggle a session-state flag (may not force rerun)
            st.session_state['_rerun_trigger'] = not st.session_state.get('_rerun_trigger', False)


# Load known encodings and names if available
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, 'rb') as f:
        known_encodings, known_names = pickle.load(f)
else:
    known_encodings = []
    known_names = []

# Function to mark attendance
def mark_attendance(name):
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')

    attendance = st.session_state.attendance_log

    # Check if attendance is already marked for the person on the same date
    if not any(entry[0] == name and entry[2] == date_str for entry in attendance):
        attendance.append([name, time_str, date_str])
        st.success(f"Attendance marked for {name}.")
        # Speak once per name per session
        if name not in st.session_state.spoken_names:
            speak(f"Attendance marked for {name}.")
            st.session_state.spoken_names.add(name)
    else:
        st.info(f"{name} is already marked present today.")

# Function to save attendance to a CSV file
def save_attendance():
    # Prevent concurrent saves
    if st.session_state.get('saving', False):
        st.info("Save already in progress. Please wait...")
        return

    attendance = st.session_state.attendance_log
    if not attendance:
        st.warning("No attendance to save.")
        return

    st.session_state['saving'] = True
    try:
        # Create DataFrame
        attendance_df = pd.DataFrame(attendance, columns=["Name", "Time", "Date"])

        # Check if the attendance file exists
        if os.path.exists(ATTENDANCE_FILE):
            existing_df = pd.read_csv(ATTENDANCE_FILE)
            attendance_df = pd.concat([existing_df, attendance_df]).drop_duplicates()

        # Save the updated DataFrame
        attendance_df.to_csv(ATTENDANCE_FILE, index=False)
        st.success("Attendance saved successfully!")
        speak("Attendance saved successfully. Thank you!")

        # Clear saved entries so a single click only saves once
        st.session_state.attendance_log = []
    except Exception as e:
        st.error(f"Error saving attendance: {e}")
    finally:
        st.session_state['saving'] = False

# Function to register a new face
def register_new_face(face_encoding, name):
    st.session_state.known_encodings.append(face_encoding)
    st.session_state.known_names.append(name)
    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump((st.session_state.known_encodings, st.session_state.known_names), f)
    st.success(f"New face registered for {name}.")

st.title("Face Recognition Attendance System")
st.write("Welcome to the Face Recognition Attendance App")

if "has_spoken_welcome" not in st.session_state:
    st.session_state.has_spoken_welcome = False

if not st.session_state.has_spoken_welcome:
    speak("Welcome to the Face Recognition Attendance App")
    st.session_state.has_spoken_welcome = True

if "known_encodings" not in st.session_state:
    st.session_state.known_encodings = known_encodings
if "known_names" not in st.session_state:
    st.session_state.known_names = known_names
if "pending_registrations" not in st.session_state:
    # Each entry: {id: str, encoding: np.array, image_bytes: bytes, added_at: float}
    st.session_state.pending_registrations = []

# Persist attendance and spoken names across reruns
if "attendance_log" not in st.session_state:
    st.session_state.attendance_log = []
if "spoken_names" not in st.session_state:
    st.session_state.spoken_names = set()

# Use streamlit-webrtc for real-time, non-blocking video processing

# Shared state between the transformer thread and the Streamlit main thread
shared_state = {
    'attendance': [],  # entries appended by transformer: [name, time_str, date_str]
    'pending': [],     # entries appended by transformer: {id, encoding, image_bytes, added_at}
    'lock': threading.Lock(),
}


class FaceTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame_counter = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_counter += 1

        # only process every Nth frame to save CPU
        PROCESS_EVERY_N = 3
        FRAME_SCALE = 0.5

        if self.frame_counter % PROCESS_EVERY_N == 0:
            try:
                small = cv2.resize(img, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
                face_locations = face_recognition.face_locations(small)
                face_encodings = face_recognition.face_encodings(small, face_locations)

                for face_encoding, face_location in zip(face_encodings, face_locations):
                    # check against known encodings
                    is_known = False
                    name = None
                    if st.session_state.get('known_encodings'):
                        try:
                            matches = face_recognition.compare_faces(st.session_state.known_encodings, face_encoding, tolerance=0.45)
                            face_distances = face_recognition.face_distance(st.session_state.known_encodings, face_encoding)
                            if len(face_distances) > 0 and matches[np.argmin(face_distances)] and face_distances[np.argmin(face_distances)] < 0.45:
                                best_match_index = np.argmin(face_distances)
                                name = st.session_state.known_names[best_match_index]
                                is_known = True
                        except Exception:
                            is_known = False

                    if is_known and name:
                        # mark attendance into shared_state (main thread will sync)
                        now = datetime.now()
                        date_str = now.strftime('%Y-%m-%d')
                        time_str = now.strftime('%H:%M:%S')
                        with shared_state['lock']:
                            shared_state['attendance'].append([name, time_str, date_str])
                        # draw box on original-sized image
                        top, right, bottom, left = [int(x / FRAME_SCALE) for x in face_location]
                        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    else:
                        # unknown: create pending registration entry
                        top_s, right_s, bottom_s, left_s = face_location
                        top, right, bottom, left = [int(top_s / FRAME_SCALE), int(right_s / FRAME_SCALE), int(bottom_s / FRAME_SCALE), int(left_s / FRAME_SCALE)]
                        h, w = img.shape[:2]
                        top_i, right_i, bottom_i, left_i = max(0, top), min(w, right), min(h, bottom), max(0, left)
                        if bottom_i > top_i and right_i > left_i:
                            try:
                                crop = img[top_i:bottom_i, left_i:right_i]
                                _, buf = cv2.imencode('.jpg', cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                                image_bytes = buf.tobytes()
                            except Exception:
                                image_bytes = None
                        else:
                            image_bytes = None

                        # dedupe by comparing encodings
                        add_pending = True
                        with shared_state['lock']:
                            for p in shared_state['pending']:
                                try:
                                    dist = np.linalg.norm(p['encoding'] - face_encoding)
                                    if dist < 0.45:
                                        add_pending = False
                                        break
                                except Exception:
                                    continue
                            if add_pending:
                                shared_state['pending'].append({
                                    'id': f"unknown_{int(time.time()*1000)}",
                                    'encoding': face_encoding,
                                    'image_bytes': image_bytes,
                                    'added_at': time.time(),
                                })

                        # draw unknown box
                        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv2.putText(img, "Unknown", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            except Exception as e:
                # avoid crashing transformer thread
                print(f"Transformer error: {e}")

        # return a VideoFrame
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# Start the webrtc streamer with the transformer
webrtc_ctx = webrtc_streamer(
    key="face-recog-stream",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=FaceTransformer,
    media_stream_constraints={"video": True, "audio": False},
)

# Sync shared_state -> st.session_state on each rerun
with shared_state['lock']:
    # move new attendance items
    if shared_state['attendance']:
        if 'attendance_log' not in st.session_state:
            st.session_state.attendance_log = []
        for entry in shared_state['attendance']:
            if not any(e[0] == entry[0] and e[2] == entry[2] for e in st.session_state.attendance_log):
                st.session_state.attendance_log.append(entry)
        shared_state['attendance'].clear()

    # move pending registrations into session_state
    if shared_state['pending']:
        if 'pending_registrations' not in st.session_state:
            st.session_state.pending_registrations = []
        # append only new ids
        existing_ids = {p['id'] for p in st.session_state.pending_registrations}
        for p in shared_state['pending']:
            if p['id'] not in existing_ids:
                st.session_state.pending_registrations.append(p)
        shared_state['pending'].clear()
    else:
        st.warning("No frames received from camera. Try restarting the camera.")

# Camera controls
col_a, col_b = st.columns(2)
with col_a:
    if st.button("Stop Camera"):
        try:
            if "video_cap" in st.session_state and st.session_state.video_cap.isOpened():
                st.session_state.video_cap.release()
        except Exception:
            pass
        st.success("Camera stopped.")
with col_b:
    if st.button("Restart Camera"):
        try:
            # release if exists then recreate
            if "video_cap" in st.session_state and st.session_state.video_cap.isOpened():
                st.session_state.video_cap.release()
        except Exception:
            pass
        st.session_state.video_cap = cv2.VideoCapture(0)
        safe_rerun()

if st.button("Save Attendance"):
    save_attendance()

if st.button("View Attendance Log"):
    if os.path.exists(ATTENDANCE_FILE):
        attendance_df = pd.read_csv(ATTENDANCE_FILE)
        st.dataframe(attendance_df)
    else:
        st.info("No attendance records found.")
