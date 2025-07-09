import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

from IPython.display import display
import ipywidgets
import ipywidgets.widgets as widgets
import traitlets

from jetbot.robot import Robot # 개별 임포트 : TensorRT 오류 우회
from jetbot.camera import Camera
from jetbot.image import bgr8_to_jpeg

import threading
import time
import cv2
import PIL.Image
import numpy as np
import json
import paho.mqtt.client as mqtt
from datetime import datetime
import pytz

import firebase_admin
from firebase_admin import credentials, db, firestore

# --- Firestore/Firebase Configuration ---
RTDB_SERVICE_ACCOUNT_KEY_PATH = 'YOUR_FIREBASE_ADMINSDK.json'
RTDB_DATABASE_URL = 'YOUR_FIREBASERTDB_PJTURL'
RTDB_APP_NAME = 'YOUR_FIREBASERTDB_APP_NAME' 

FIRESTORE_SERVICE_ACCOUNT_KEY_PATH = 'YOUR_FIRESTORE_ADMINSDK.json'
FIRESTORE_APP_NAME = 'YOUR_FIRESTORE_APP_NAME'

DISTANCE_DB_PATH = 'YOUR_RTDB_DIST_PATH'
HUMID_DB_PATH = 'YOUR_RTDB_HUMID_PATH'
TEMP_DB_PATH = 'YOUR_RTDB_TEMP_PATH'

FIRESTORE_COLLECTION_NAME = 'YOUR_FIRESTOREDB_COLLECTION_NAME' 

RTDB_APP_INITIALIZED = False
FIRESTORE_APP_INITIALIZED = False
korea_timezone = pytz.timezone("Asia/Seoul")

fs_client = None # Firestore 클라이언트 (전역 변수)

# --- Firebase Functions ---
def initialize_firebase_apps():
    global RTDB_APP_INITIALIZED, FIRESTORE_APP_INITIALIZED, fs_client

    try: # RTDB 앱 초기화
        try:
            firebase_admin.get_app(name=RTDB_APP_NAME)
            print(f"INFO: Firebase app '{RTDB_APP_NAME}' already exists.")
            RTDB_APP_INITIALIZED = True
        except ValueError:
            cred_rtdb = credentials.Certificate(RTDB_SERVICE_ACCOUNT_KEY_PATH)
            firebase_admin.initialize_app(cred_rtdb, {
                'databaseURL': RTDB_DATABASE_URL
            }, name=RTDB_APP_NAME)
            print(f"Firebase App '{RTDB_APP_NAME}' (RTDB) initialized successfully.")
            RTDB_APP_INITIALIZED = True
    except FileNotFoundError:
        print(f"ERROR: RTDB key file not found at '{RTDB_SERVICE_ACCOUNT_KEY_PATH}'.")
    except Exception as e:
        print(f"An unexpected error occurred during RTDB app initialization: {e}")

    try: # Firestore App 초기화
        try:
            firebase_admin.get_app(name=FIRESTORE_APP_NAME)
            print(f"INFO: Firebase app '{FIRESTORE_APP_NAME}' already exists.")
            FIRESTORE_APP_INITIALIZED = True
        except ValueError:
            cred_fs = credentials.Certificate(FIRESTORE_SERVICE_ACCOUNT_KEY_PATH)
            firebase_admin.initialize_app(cred_fs, name=FIRESTORE_APP_NAME)
            print(f"Firebase App '{FIRESTORE_APP_NAME}' (Firestore) initialized successfully.")
            FIRESTORE_APP_INITIALIZED = True
    except FileNotFoundError:
        print(f"ERROR: Firestore key file not found at '{FIRESTORE_SERVICE_ACCOUNT_KEY_PATH}'.")
    except Exception as e:
        print(f"An unexpected error occurred during Firestore app initialization: {e}")

    # Firestore 클라이언트 가져오기
    if FIRESTORE_APP_INITIALIZED:
        try:
            firestore_app = firebase_admin.get_app(name=FIRESTORE_APP_NAME)
            fs_client = firestore.client(app=firestore_app)
            print("Firestore client obtained successfully.")
        except Exception as e:
            print(f"Error obtaining Firestore client: {e}")

    return RTDB_APP_INITIALIZED and FIRESTORE_APP_INITIALIZED

# --- Firebase RTDB 데이터 가져오기 ---
def get_firebase_data_from_rtdb(db_path):
    if not RTDB_APP_INITIALIZED: return None
    try:
        rtdb_app = firebase_admin.get_app(name=RTDB_APP_NAME)
        ref = db.reference(db_path, app=rtdb_app)
        return ref.get()
    except Exception as e:
        print(f"Error fetching data from {db_path}: {e}"); return None

# --- Firebase RTDB 개별 데이터 가져오기 ---
def get_firebase_distance():
    val = get_firebase_data_from_rtdb(DISTANCE_DB_PATH)
    return int(val) if isinstance(val, (int, float)) else None

def get_firebase_humidity():
    val = get_firebase_data_from_rtdb(HUMID_DB_PATH)
    return float(val) if isinstance(val, (int, float)) else None

def get_firebase_temperature():
    val = get_firebase_data_from_rtdb(TEMP_DB_PATH)
    return float(val) if isinstance(val, (int, float)) else None

# --- Firestore Database 데이터 업로드 ---
def upload_to_firestore(collection_name, document_id, data):
    """지정된 컬렉션과 문서 ID로 데이터를 업로드(덮어쓰기)합니다."""
    global fs_client
    if not fs_client:
        print("Firestore client not initialized. Cannot upload data.")
        return
    try:
        fs_client.collection(collection_name).document(document_id).set(data)
    except Exception as e:
        print(f"Error setting data to Firestore: {e}")

# --- Initialize Firebase ---
initialize_firebase_apps()

# --- Robot & Camera & Model Setup ---
robot = Robot()
camera = Camera.instance(width=224, height=224)

model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 2)
try:
    model.load_state_dict(torch.load('../road_following/best_steering_model_xy_test.pth'))
except FileNotFoundError:
    print("ERROR: Model file not found. ../road_following/best_steering_model_xy_test.pth")
    exit()

device = torch.device('cuda')
model = model.to(device)
model = model.eval().half()

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()
print('Model load success')

# --- Color Definitions ---
colors = [
    {'name': 'red', 'lower': np.array([0, 88, 146]), 'upper': np.array([10, 168, 226])},
    {'name': 'blue', 'lower': np.array([95, 150, 140]), 'upper': np.array([105, 255, 255])},
    {'name': 'purple', 'lower': np.array([126, 50, 80]), 'upper': np.array([145, 180, 200])},
    {'name': 'yellow', 'lower': np.array([20, 135, 163]), 'upper': np.array([50, 170, 195])},
    {'name': 'orange', 'lower': np.array([13, 160, 150]), 'upper': np.array([24, 255, 255])},
]

# --- Data Storage ---
color_data = {
    'red': {'temp': [], 'humid': []}, 'blue': {'temp': [], 'humid': []},
    'purple': {'temp': [], 'humid': []}, 'yellow': {'temp': [], 'humid': []},
    'orange': {'temp': [], 'humid': []},
}

# --- UI Widgets Setup ---
lbl1 = ipywidgets.Label(value="Status :")
areaAlbl = ipywidgets.Label(value="Stopped")
hbox1 = widgets.HBox([lbl1, areaAlbl])
lbl2 = ipywidgets.Label(value="Found :")
areaBlbl = ipywidgets.Label(value="None")
hbox2 = widgets.HBox([lbl2, areaBlbl])
lbl3 = ipywidgets.Label(value="Details:")
flaglbl = ipywidgets.Label(value="-")
hbox3 = widgets.HBox([lbl3, flaglbl])
vbox1 = widgets.VBox([hbox1, hbox2, hbox3])

image_widget = ipywidgets.Image(format='jpeg', width=224, height=224)
x_slider = ipywidgets.FloatSlider(min=-1.0, max=1.0, description='x', value=0.0)
y_slider = ipywidgets.FloatSlider(min=0, max=1.0, orientation='vertical', description='y', value=0.0)
steering_slider = ipywidgets.FloatSlider(min=-1.0, max=1.0, description='steering', value=0.0)
speed_slider = ipywidgets.FloatSlider(min=0, max=1.0, orientation='vertical', description='speed', value=0.0)
vbox2 = widgets.VBox([image_widget, x_slider, steering_slider], layout=widgets.Layout(align_self='center'))
hbox4 = widgets.HBox([vbox2, y_slider, speed_slider], layout=widgets.Layout(align_self='center'))

startBtn = widgets.Button(description="Start", button_style='info')
lbl41 = ipywidgets.Label(value="Current Color:")
goallbl = ipywidgets.Label(value="None")
hbox5 = widgets.HBox([startBtn, lbl41, goallbl])

lbl50 = ipywidgets.Label(value="Manual Controller")
button_layout = widgets.Layout(width='100px', height='80px', align_self='center')
stop_button = widgets.Button(description='stop', button_style='danger', layout=button_layout)
forward_button = widgets.Button(description='forward', layout=button_layout)
backward_button = widgets.Button(description='backward', layout=button_layout)
left_button = widgets.Button(description='left', layout=button_layout)
right_button = widgets.Button(description='right', layout=button_layout)
middle_box = widgets.HBox([left_button, stop_button, right_button], layout=widgets.Layout(align_self='center'))
controls_box = widgets.VBox([lbl50, forward_button, middle_box, backward_button])

lbl51 = ipywidgets.Label(value="Auto Controller")
speed_gain_slider = ipywidgets.FloatSlider(min=0.0, max=1.0, step=0.01, value=0.17, description='speed gain',disabled = False)
steering_gain_slider = ipywidgets.FloatSlider(min=0.0, max=1.0, step=0.01, value=0.2, description='steering gain',disabled = False)
steering_dgain_slider = ipywidgets.FloatSlider(min=0.0, max=0.5, step=0.001, value=0.0, description='steering kd',disabled = False)
steering_bias_slider = ipywidgets.FloatSlider(min=-0.3, max=0.3, step=0.01, value=0.0, description='steering bias',disabled = False)
vbox3 = widgets.VBox([lbl51,speed_gain_slider, steering_gain_slider, steering_dgain_slider, steering_bias_slider])
hbox6 = widgets.HBox([controls_box, vbox3], layout=widgets.Layout(align_self='center'))

camera_link = traitlets.dlink((camera, 'value'), (image_widget, 'value'), transform=bgr8_to_jpeg)

# --- Manual Control Functions ---
def stop_robot_action(change): robot.stop()
def step_forward(change): robot.forward(0.4); time.sleep(0.5); robot.stop()
def step_backward(change): robot.backward(0.4); time.sleep(0.5); robot.stop()
def step_left(change): robot.left(0.3); time.sleep(0.5); robot.stop()
def step_right(change): robot.right(0.3); time.sleep(0.5); robot.stop()

stop_button.on_click(stop_robot_action)
forward_button.on_click(step_forward)
backward_button.on_click(step_backward)
left_button.on_click(step_left)
right_button.on_click(step_right)

# --- Global Thread Variables ---
roadFinding = None
goalFinding = None
sensingThread = None
mqtt_client = None
current_agv_mode = "manual"

# --- MQTT Configuration ---
MQTT_BROKER_HOST = "YOUR_BROKER_HOST_IP"
MQTT_BROKER_PORT = "YOUR_BROKER_PORT_NUM"
COMMAND_TOPIC = "YOUR_COMMAND_TOPIC"
SENSING_TOPIC = "YOUR_SENSING_TOPIC"

# --- MQTT Callbacks ---
def on_connect_agv(client, userdata, flags, rc):
    global current_agv_mode
    reason_code = rc.value if hasattr(rc, 'value') else rc
    
    if reason_code == 0:
        print("AGV Controller: Connected to MQTT Broker OK")
        client.subscribe(COMMAND_TOPIC, qos=1)
        print(f"AGV Controller: Subscribed to {COMMAND_TOPIC}")
        current_agv_mode = "manual"
        robot.stop()
    else:
        print(f"AGV Controller: Connection failed. Code: {reason_code}")

def on_message_agv(client, userdata, msg):
    global current_agv_mode, roadFinding, goalFinding
    try:
        payload = json.loads(msg.payload.decode())
        command = payload.get("cmd_string")
        print(f"AGV Controller Received MQTT Command: {command}")

        if current_agv_mode == "auto" and command not in ["auto_stop", "mid", "stop", "exit"]:
            print(f"AGV: Ignoring manual command '{command}' while in Auto Mode.")
            return

        if command == "go": robot.forward(0.4); time.sleep(0.5); robot.stop()
        elif command == "back": robot.backward(0.4); time.sleep(0.5); robot.stop()
        elif command == "left": robot.left(0.3); time.sleep(0.5); robot.stop()
        elif command == "right": robot.right(0.3); time.sleep(0.5); robot.stop()
        elif command == "mid" or command == "stop":
            if current_agv_mode == "auto":
                current_agv_mode = "manual"
                stop_all_threads(None) 
                startBtn.description = "Start"; startBtn.button_style = "info"
            else: robot.stop()
            print("AGV: Stopped (via MQTT)")
        elif command == "auto_start":
            if current_agv_mode == "manual":
                current_agv_mode = "auto"
                start_all_threads(None)
                startBtn.description = "Stop"; startBtn.button_style = "warning"
            print("AGV: Auto Mode Started (via MQTT)")
        elif command == "auto_stop":
            if current_agv_mode == "auto":
                current_agv_mode = "manual"
                stop_all_threads(None)
                startBtn.description = "Start"; startBtn.button_style = "info"
            print("AGV: Auto Mode Stopped (via MQTT)")
        elif command == "exit":
            print("AGV: Received Exit command.")
            stop_all_threads(None)
            if mqtt_client: mqtt_client.disconnect(); mqtt_client.loop_stop()
        else: print(f"AGV Controller: Unknown command '{command}'")
    except Exception as e:
        print(f"AGV Controller Error during message processing: {e}")

# --- SensingThread Class ---
class SensingThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.th_flag = True
        self.publish_interval = 1.0

    def run(self):
        global mqtt_client, areaAlbl, areaBlbl, flaglbl, current_agv_mode, korea_timezone, FIRESTORE_COLLECTION_NAME, fs_client
        while self.th_flag:
            try:
                # 데이터 가져오기 : From Firebase RTDB
                now = datetime.now(korea_timezone)
                current_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
                temp_val = get_firebase_temperature()
                humid_val = get_firebase_humidity()
                dist_val = get_firebase_distance() 

                # 충돌 여부 판단
                is_collision = (dist_val is not None and dist_val <= 30)

                # 데이터 업로드 : To Firestore Database
                document_id = now.strftime("%Y-%m-%d_%H-%M-%S-%f")
                firestore_payload = {
                    "time": current_time_str,
                    "temperature": temp_val if temp_val is not None else None,
                    "humidity": humid_val if humid_val is not None else None,
                    "is_collision_detected": is_collision 
                }
                
                if fs_client: 
                    upload_to_firestore(FIRESTORE_COLLECTION_NAME, document_id, firestore_payload)

                # MQTT 연결 확인 및 publish
                if mqtt_client and mqtt_client.is_connected():
                    payload = {
                        "time": current_time_str, "status": areaAlbl.value, "found": areaBlbl.value,
                        "details": flaglbl.value, "mode": current_agv_mode,
                        "temperature": temp_val if temp_val is not None else "N/A",
                        "humidity": humid_val if humid_val is not None else "N/A",
                        "num1": temp_val if temp_val is not None else 0.0,
                        "num2": humid_val if humid_val is not None else 0.0,
                        "is_finish": 1 if current_agv_mode == 'auto' else 0,
                        "manual_mode": current_agv_mode,
                        "is_collision": is_collision 
                    }
                    mqtt_client.publish(SENSING_TOPIC, json.dumps(payload), qos=0)

            except Exception as e:
                print(f"Error in SensingThread run loop: {e}")
            
            time.sleep(self.publish_interval)

    def stop(self):
        self.th_flag = False

# --- ColorAreaDetector Class ---
class ColorAreaDetector(threading.Thread):
    def __init__(self):
        super().__init__()
        self.th_flag = True
        self.imageInput = None
        self.min_radius_threshold = 10

    def run(self):
        global camera
        while self.th_flag:
            if camera.value is None: time.sleep(0.01); continue
            self.imageInput = camera.value.copy()
            hsv = cv2.cvtColor(self.imageInput, cv2.COLOR_BGR2HSV)
            hsv = cv2.blur(hsv, (5, 5))
            found_colors_list = []
            for color_spec in colors:
                mask = cv2.inRange(hsv, color_spec['lower'], color_spec['upper'])
                mask = cv2.erode(mask, None, iterations=2)
                mask = cv2.dilate(mask, None, iterations=2)
                contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    ((box_x, box_y), radius) = cv2.minEnclosingCircle(c)
                    if radius > self.min_radius_threshold:
                        found_colors_list.append({'name': color_spec['name'], 'x': int(box_x), 'y': int(box_y), 'radius': radius})

            if found_colors_list:
                largest_color = max(found_colors_list, key=lambda item: item['radius'])
                color_name = largest_color['name']
                current_temp = get_firebase_temperature()
                current_humid = get_firebase_humidity()
                if current_temp is not None and current_humid is not None:
                    if color_name in color_data:
                        color_data[color_name]['temp'].append(current_temp)
                        color_data[color_name]['humid'].append(current_humid)
                goallbl.value = color_name.upper()
                areaAlbl.value = "Color Detected!"
                areaBlbl.value = color_name
                flaglbl.value = f"At ({largest_color['x']}, {largest_color['y']}), R={largest_color['radius']:.1f}"
                cv2.circle(self.imageInput, (largest_color['x'], largest_color['y']), int(largest_color['radius']), (0, 255, 255), 2)
                cv2.putText(self.imageInput, color_name, (largest_color['x'] - 20, largest_color['y'] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                goallbl.value = "None"; areaAlbl.value = "Searching..."; areaBlbl.value = "None"; flaglbl.value = "-"
                if self.imageInput is not None:
                    cv2.putText(self.imageInput, "Searching...", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
            
            if self.imageInput is not None:
                image_widget.value = bgr8_to_jpeg(self.imageInput)
            time.sleep(0.1)

    def stop(self): self.th_flag = False

# --- RobotMoving Class ---
class RobotMoving(threading.Thread):
    def __init__(self):
        super().__init__()
        self.th_flag = True
        self.angle = 0.0
        self.angle_last = 0.0

    def run(self):
        global camera, robot, model
        while self.th_flag:
            if current_agv_mode == "manual": time.sleep(0.1); continue
            if camera.value is None: time.sleep(0.01); continue
            image = camera.value
            xy = model(self.preprocess(image)).detach().float().cpu().numpy().flatten()
            x = xy[0]; y = (0.5 - xy[1]) / 2.0
            x_slider.value = x; y_slider.value = y
            firebase_dist = get_firebase_distance()
            perform_normal_driving = True
            if firebase_dist is not None:
                if firebase_dist <= 30:
                    print(f"INFO: Distance ({firebase_dist}) <= 30. Stopping.")
                    robot.stop(); speed_slider.value = 0.0; steering_slider.value = 0.0
                    perform_normal_driving = False
            
            if perform_normal_driving:
                effective_speed = speed_gain_slider.value
                speed_slider.value = effective_speed
                self.angle = np.arctan2(x, y) 
                pid = self.angle * steering_gain_slider.value + (self.angle - self.angle_last) * steering_dgain_slider.value
                self.angle_last = self.angle
                current_steering_value = pid + steering_bias_slider.value
                steering_slider.value = current_steering_value
                left_motor_val = max(min(effective_speed + current_steering_value, 1.0), 0.0)
                right_motor_val = max(min(effective_speed - current_steering_value, 1.0), 0.0)
                robot.left_motor.value = left_motor_val
                robot.right_motor.value = right_motor_val
            
            time.sleep(0.05)
        
        robot.stop()

    def preprocess(self, image_array):
        image = PIL.Image.fromarray(image_array)
        image = transforms.functional.to_tensor(image).to(device).half()
        image.sub_(mean[:, None, None]).div_(std[:, None, None])
        return image[None, ...]

    def stop(self): self.th_flag = False; time.sleep(0.1); robot.stop()

# --- Thread Control Functions ---
def start_all_threads(change):
    global camera_link, goalFinding, roadFinding, sensingThread, current_agv_mode

    if change is not None:
        current_agv_mode = "auto"
        print("AGV: UI Start button pressed, switching to Auto Mode.")
        startBtn.button_style = "warning"
        startBtn.description = "Stop"

    if camera_link: camera_link.unlink(); camera_link = None
    if goalFinding is None or not goalFinding.is_alive():
        goalFinding = ColorAreaDetector(); goalFinding.start()
    if roadFinding is None or not roadFinding.is_alive():
        roadFinding = RobotMoving(); roadFinding.start()
    if sensingThread is None or not sensingThread.is_alive():
        sensingThread = SensingThread(); sensingThread.start()

def stop_all_threads(change):
    global camera_link, goalFinding, roadFinding, sensingThread, current_agv_mode

    if change is not None:
        current_agv_mode = "manual"
        print("AGV: UI Stop button pressed, switching to Manual Mode.")
        startBtn.button_style = "info"
        startBtn.description = "Start"

    if roadFinding: roadFinding.stop(); roadFinding.join(); roadFinding = None
    if goalFinding: goalFinding.stop(); goalFinding.join(); goalFinding = None
    if sensingThread: sensingThread.stop(); sensingThread.join(); sensingThread = None

    robot.stop() 
    
    if not camera_link:
        try:
            camera_link = traitlets.dlink((camera, 'value'), (image_widget, 'value'), transform=bgr8_to_jpeg)
        except Exception as e:
            print(f"Error relinking camera: {e}")

# --- UI Button Click Handler ---
def handle_start_stop_button(change):
    if startBtn.description == "Start":
        start_all_threads(change)
    else:
        stop_all_threads(change)

startBtn.on_click(handle_start_stop_button)

# --- Initialize MQTT Client ---
def init_mqtt_client_agv():
    global mqtt_client
    if mqtt_client is None:
        try:
            mqtt_client = mqtt.Client() 
            mqtt_client.on_connect = on_connect_agv
            mqtt_client.on_message = on_message_agv
            mqtt_client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
            mqtt_client.loop_start()
            print("AGV Controller: MQTT client initiated and loop started.")
        except Exception as e:
            print(f"AGV Controller: Failed to start MQTT client: {e}")
            mqtt_client = None 
    else:
        print("AGV Controller: MQTT client already initialized.")

init_mqtt_client_agv()

# --- Display UI ---
display(vbox1, hbox4, hbox5, hbox6)
print("INFO: Script setup complete. Displaying UI.")