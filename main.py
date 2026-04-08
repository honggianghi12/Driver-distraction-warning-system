import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
import requests
import threading
from queue import Queue
import dlib

print(" Khởi động hệ thống ESP32-CAM Eye Detection...")

# ====================
# CẤU HÌNH
# ====================
ESP32_CAM_URL = "http://172.20.10.13"  # Thay IP thực tế
ESP32_CONTROL_URL = "http://172.20.10.14"
CLOSED_EYE_THRESHOLD = 0.5
ALERT_DURATION = 3  # thời gian cảnh báo tối thiêu
ALERT_INTERVAL = 3  # ngưỡng cảnh báo để gửi json tiếp


# LOAD MODEL VÀ DETECTOR

try:
    model = load_model('eye_model.h5')
    print(" Đã tải model eye_model.h5")
except:
    print(" Không thể tải model eye_model.h5")
    exit()

# Chỉ dùng Haar để phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load dlib cho phát hiện mắt
try:
    dlib_predictor = dlib.shape_predictor(
        "shape_predictor_68_face_landmarks.dat")
    print(" Đã tải dlib landmark predictor")
except Exception as e:
    print(f" Lỗi tải dlib: {e}")
    exit()

# ====================
# BIẾN TOÀN CỤC
# ====================
eyes_closed_time = None
alert_active = False
alert_serial_sent = False
frame_queue = Queue(maxsize=2)
last_frame_time = time.time()
fps = 0
frame_count = 0
last_alert_time = 0

# HÀM GIAO TIẾP VỚI ESP32 CONTROL

def send_alert_to_esp32(alert_type="start"):
    """Gửi cảnh báo đến ESP32 control"""
    try:
        if alert_type == "start":
            response = requests.post(
                f"{ESP32_CONTROL_URL}/alert",
                json={"active": True},
                timeout=2
            )
            print(" Đã gửi cảnh báo buồn ngủ đến ESP32")
        else:
            response = requests.post(
                f"{ESP32_CONTROL_URL}/alert",
                json={"active": False},
                timeout=2
            )
            print(" Đã gửi tín hiệu bình thường đến ESP32")

    except Exception as e:
        print(f" Lỗi kết nối ESP32 control: {e}")

# HÀM LẤY FRAME

def get_esp32_frame_fast():
    try:
        response = requests.get(f"{ESP32_CAM_URL}/capture", timeout=2)
        if response.status_code == 200:
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return frame
    except Exception as e:
        return None
    return None


def esp32_frame_grabber():
    while True:
        frame = get_esp32_frame_fast()
        if frame is not None:
            if not frame_queue.full():
                frame_queue.put(frame)


# HÀM XỬ LÝ MẮT - HAAR FACE + DLIB EYES



def preprocess_eye(eye_img):
    """Tiền xử lý ảnh mắt"""
eye_img = cv2.resize(eye_img, (24, 24))
    eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    eye_img = eye_img.astype('float32') / 255.0
    eye_img = eye_img.reshape(1, 24, 24, 1)
    return eye_img



def predict_eye_state(eye_img):
    """Dự đoán trạng thái mắt"""
    processed_eye = preprocess_eye(eye_img)
    prediction = model.predict(processed_eye, verbose=0)
    return prediction[0][0]


def get_eye_region_from_landmarks(landmarks, eye_points):
    points = []
    for i in eye_points:
        point = landmarks.part(i)
        points.append((point.x, point.y))

    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Mở rộng vùng mắt một chút
    expand = 5
    x_min = max(0, x_min - expand)
    y_min = max(0, y_min - expand)
    x_max = x_max + expand
    y_max = y_max + expand

    return (x_min, y_min, x_max - x_min, y_max - y_min)


def process_eye_detection_haar_dlib(frame):
    """Xử lý phát hiện mắt: Haar face + Dlib eyes + AI model"""
    global eyes_closed_time, alert_active, alert_serial_sent, last_alert_time

    display_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    current_eye_state = "searching"
    face_detected = False
    eye_predictions = []

    # BƯỚC 1: HAAR PHÁT HIỆN KHUÔN MẶT
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=4,
        minSize=(80, 80),
        maxSize=(300, 300)
    )

    for (x, y, w, h) in faces:
        face_detected = True

        # Vẽ khuôn mặt
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        #  DLIB PHÁT HIỆN MẮT TỪ KHUÔN MẶT
        try:
            # Chuyển Haar rectangle sang dlib rectangle
            dlib_rect = dlib.rectangle(left=x, top=y, right=x+w, bottom=y+h)

            # Dlib lấy landmarks
            landmarks = dlib_predictor(gray, dlib_rect)

            #  TRÍCH XUẤT VÙNG MẮT TỪ LANDMARKS
            # Mắt trái (landmarks 36-41)
            left_eye_region = get_eye_region_from_landmarks(
                landmarks, range(36, 42))
            # Mắt phải (landmarks 42-47)
            right_eye_region = get_eye_region_from_landmarks(
                landmarks, range(42, 48))

            # Xử lý mắt trái
            ex, ey, ew, eh = left_eye_region
            try:
                eye_img = frame[ey:ey+eh, ex:ex+ew]
                eye_prob = predict_eye_state(eye_img)
                # lưu xác suất để trung bình 2 mắt.
                eye_predictions.append(eye_prob)
                # Vẽ khung mắt
                eye_color = (255, 0, 0) if eye_prob > CLOSED_EYE_THRESHOLD else (
                    0, 255, 255)
                cv2.rectangle(display_frame, (ex, ey),
(ex+ew, ey+eh), eye_color, 2)
            except Exception as e:
                print(f"Lỗi xử lý mắt trái: {e}")

            # Xử lý mắt phải
            ex, ey, ew, eh = right_eye_region
            try:
                eye_img = frame[ey:ey+eh, ex:ex+ew]
                eye_prob = predict_eye_state(eye_img)
                eye_predictions.append(eye_prob)

                # Vẽ khung mắt
                eye_color = (255, 0, 0) if eye_prob > CLOSED_EYE_THRESHOLD else (
                    0, 255, 255)
                cv2.rectangle(display_frame, (ex, ey),
                              (ex+ew, ey+eh), eye_color, 2)
            except Exception as e:
                print(f"Lỗi xử lý mắt phải: {e}")

        except Exception as e:
            print(f"Lỗi dlib: {e}")
            continue

        # PHÂN TÍCH TRẠNG THÁI MẮT
        if len(eye_predictions) >= 1:
            avg_prob = np.mean(eye_predictions)  # Lấy trung bình 2 mắt

            if avg_prob > CLOSED_EYE_THRESHOLD:
                current_eye_state = "open"
                eyes_closed_time = None
                alert_active = False
                alert_serial_sent = False
                face_color = (0, 255, 0)
            else:
                current_eye_state = "closed"
                if eyes_closed_time is None:
                    eyes_closed_time = time.time()

                closed_duration = time.time() - eyes_closed_time
                face_color = (
                    0, 255, 255) if closed_duration < ALERT_DURATION else (0, 0, 255)
                alert_active = closed_duration >= ALERT_DURATION
        else:
            current_eye_state = "no_eyes"
            face_color = (255, 0, 0)

        # Hiển thị trạng thái trên khung mặt
        cv2.putText(display_frame, current_eye_state.upper(), (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, face_color, 2)

        # Hiển thị thời gian nhắm mắt
        if current_eye_state == "closed" and eyes_closed_time is not None:
            closed_duration = time.time() - eyes_closed_time
            cv2.putText(display_frame, f"Closed: {closed_duration:.1f}s",
                        (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)

        # Chỉ xử lý 1 khuôn mặt đầu tiên
        break

    # XỬ LÝ CẢNH BÁO
    if alert_active:
        current_time = time.time()
        cv2.putText(display_frame, "ALERT! SLEEP DETECTED", (10, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if (current_time - last_alert_time) >= ALERT_INTERVAL:
            send_alert_to_esp32("start")
            last_alert_time = current_time
            print("CẢNH BÁO: Đã kích hoạt hệ thống báo động!")
    else:
        if alert_serial_sent:
            send_alert_to_esp32("stop")
            alert_serial_sent = False
            last_alert_time = 0
return display_frame, current_eye_state, face_detected


# HÀM CHÍNH



def main():
    global last_frame_time, fps, frame_count

    # Kiểm tra kết nối ESP32
    print("\n Kiểm tra kết nối ESP32-CAM...")
    test_frame = get_esp32_frame_fast()
    if test_frame is None:
        print("Không thể kết nối đến ESP32-CAM!")
        print("   IP hiện tại:", ESP32_CAM_URL)
        return

    print(f"Kết nối ESP32-CAM thành công!")
    print(f"Kích thước frame: {test_frame.shape[1]}x{test_frame.shape[0]}")

    # Khởi chạy luồng grab frame
    # deamon True nghĩa là thread sẽ tự dừng khi main thread kết thúc.
    grabber_thread = threading.Thread(target=esp32_frame_grabber, daemon=True)
    grabber_thread.start()

    print("Bắt đầu xử lý ...")

    last_fps_time = time.time()
    consecutive_failures = 0

    while True:
        # Lấy frame từ queue
        current_frame = None
        if not frame_queue.empty():
            current_frame = frame_queue.get()
            consecutive_failures = 0
        else:
            # Fallback nhanh
            current_frame = get_esp32_frame_fast()
            if current_frame is None:
                consecutive_failures += 1
                if consecutive_failures > 10:
                    print("Mất kết nối ESP32-CAM...")
                    consecutive_failures = 0

        if current_frame is not None:
            # Tính FPS
            current_time = time.time()
            frame_count += 1

            # Hiển thị FPS mỗi giây
            if current_time - last_fps_time >= 1.0:
                fps = frame_count / (current_time - last_fps_time)
                frame_count = 0
                last_fps_time = current_time

            # Xử lý frame với Haar + Dlib
            processed_frame, eye_state, face_detected = process_eye_detection_haar_dlib(
                current_frame)

            # Hiển thị thông tin
            cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow('ESP32-CAM Eye Detection - Haar + Dlib',
                       processed_frame)

            # Log FPS mỗi 3 giây
            if int(time.time()) % 3 == 0:
                print(f"FPS: {fps:.1f} | State: {eye_state}")

        # Thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    send_alert_to_esp32("stop")
    print("Đã thoát chương trình.")


if __name__ == "__main__":
    main()