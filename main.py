import streamlit as st
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

    # Run the speaking function in a separate thread
    thread = threading.Thread(target=run_speak, daemon=True)
    thread.start()

# Note: persist attendance and spoken names in Streamlit session state

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
    attendance = st.session_state.attendance_log
    if not attendance:
        st.warning("No attendance to save.")
        return

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

# Function to register a new face
def register_new_face(face_encoding, name):
    known_encodings.append(face_encoding)
    known_names.append(name)
    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump((known_encodings, known_names), f)
    st.success(f"New face registered for {name}.")


# Robust rerun helper to support multiple Streamlit versions
def safe_rerun():
    """Try to programmatically rerun the Streamlit script across versions.

    Order of attempts:
    1. Call `st.experimental_rerun()` if available.
    2. Try importing and raising known runtime rerun exceptions.
    3. Fallback to `st.stop()` to halt execution (best-effort fallback).
    """
    # Preferred public API
    rerun_fn = getattr(st, "experimental_rerun", None)
    if callable(rerun_fn):
        try:
            rerun_fn()
            return
        except Exception:
            # If it exists but fails, continue to other fallbacks
            pass

    # Try common internal RerunException locations (best-effort)
    try:
        from streamlit.runtime.scriptrunner import RerunException
        raise RerunException()
    except Exception:
        try:
            from streamlit.script_runner import RerunException
            raise RerunException()
        except Exception:
            # Final fallback: stop the script (safe)
            try:
                st.stop()
            except Exception:
                # As a last resort, do nothing
                return

# Streamlit UI
st.title("Face Recognition Attendance System")
st.write("Welcome to the Face Recognition Attendance System")

# Use session state to ensure the welcome message is spoken only once
if "has_spoken_welcome" not in st.session_state:
    st.session_state.has_spoken_welcome = False

if not st.session_state.has_spoken_welcome:
    speak("Welcome to the Face Recognition Attendance System")
    st.session_state.has_spoken_welcome = True

if "known_encodings" not in st.session_state:
    st.session_state.known_encodings = known_encodings
if "known_names" not in st.session_state:
    st.session_state.known_names = known_names
if "pending_registrations" not in st.session_state:
    # Each entry: {id: str, encoding: np.array, image_bytes: bytes, added_at: float}
    st.session_state.pending_registrations = []

if "attendance_log" not in st.session_state:
    st.session_state.attendance_log = []
if "spoken_names" not in st.session_state:
    st.session_state.spoken_names = set()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Error: Could not open camera.")
else:
    ret, frame = cap.read()

    if ret:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if face_encodings:
            for face_encoding, face_location in zip(face_encodings, face_locations):
                # Use a stricter tolerance for face matching
                TOLERANCE = 0.45
                matches = face_recognition.compare_faces(st.session_state.known_encodings, face_encoding, tolerance=TOLERANCE)
                face_distances = face_recognition.face_distance(st.session_state.known_encodings, face_encoding)


                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index] and face_distances[best_match_index] < TOLERANCE:
                        name = st.session_state.known_names[best_match_index]
                        mark_attendance(name)
                        # Draw a rectangle and label for recognized face
                        top, right, bottom, left = face_location
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            name,
                            (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2,
                        )
                    else:
                        # Unknown face: add to pending registrations (debounced)
                        # Crop the face from the frame to show to the user
                        top, right, bottom, left = face_location
                        # ensure coordinates are ints and within frame bounds
                        h, w = frame.shape[:2]
                        top_i, right_i, bottom_i, left_i = map(int, [max(0, top), min(w, right), min(h, bottom), max(0, left)])
                        try:
                            crop = frame[top_i:bottom_i, left_i:right_i]
                        except Exception:
                            crop = None

                        # Encode crop to bytes so it can be stored in session state
                        image_bytes = None
                        if crop is not None and crop.size > 0:
                            _, img_buf = cv2.imencode('.jpg', cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                            image_bytes = img_buf.tobytes()

                        # Create an id for this pending registration
                        pending_id = f"unknown_{int(time.time()*1000)}"

                        # Avoid adding many duplicates: compare to existing pending by distance
                        add_pending = True
                        for p in st.session_state.pending_registrations:
                            try:
                                dist = np.linalg.norm(p['encoding'] - face_encoding)
                                if dist < 0.45:
                                    add_pending = False
                                    break
                            except Exception:
                                continue

                        if add_pending:
                            st.session_state.pending_registrations.append({
                                'id': pending_id,
                                'encoding': face_encoding,
                                'image_bytes': image_bytes,
                                'added_at': time.time(),
                            })

                        # Draw a rectangle for unknown face
                        top, right, bottom, left = face_location
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv2.putText(
                            frame,
                            "Unknown",
                            (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2,
                        )

            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Processed Image")

            # Show UI for any pending registrations
            if st.session_state.pending_registrations:
                st.markdown("### Pending Registrations")
                # Render each pending registration with its image and a name input + register button
                for p in list(st.session_state.pending_registrations):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        if p.get('image_bytes'):
                            st.image(p['image_bytes'], caption="Unknown Face", use_column_width=True)
                        else:
                            st.write("No preview available")
                    with col2:
                        name_key = f"name_{p['id']}"
                        btn_key = f"btn_{p['id']}"
                        new_name = st.text_input("Enter name to register", key=name_key)
                        if st.button("Register & Mark Attendance", key=btn_key):
                            if new_name:
                                register_new_face(p['encoding'], new_name)
                                mark_attendance(new_name)
                                # Immediately persist the newly marked attendance to CSV
                                save_attendance()
                                st.session_state.pending_registrations = [x for x in st.session_state.pending_registrations if x['id'] != p['id']]
                                # Programmatic rerun with compatibility fallback
                                safe_rerun()
                            else:
                                st.warning("Please enter a name before registering.")
        else:
            st.warning("No faces detected. Please try again.")

        cap.release()

if st.button("Save Attendance"):
    save_attendance()

if st.button("View Attendance Log"):
    if os.path.exists(ATTENDANCE_FILE):
        attendance_df = pd.read_csv(ATTENDANCE_FILE)
        st.dataframe(attendance_df)
    else:
        st.info("No attendance records found.")

