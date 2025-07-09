# -*- coding: utf-8 -*-
import sys
import os
import json
import time
import io  # 이미지 버퍼링

import paho.mqtt.client as mqtt
import firebase_admin
from firebase_admin import credentials, firestore
from openai import OpenAI

# GUI 및 시각화 관련 라이브러리
from PySide2.QtWidgets import (QApplication, QMainWindow, QMessageBox, 
                             QLabel, QVBoxLayout, QWidget) # 시각화 표시용
from PySide2.QtCore import Slot, QThread, Signal
from PySide2.QtGui import QFont, QPixmap # 이미지 표시용
import pandas as pd
import matplotlib.pyplot as plt
from ui_form import Ui_MainWindow # 사용자 UI 파일 (반드시 존재해야 함)

OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
FIREBASE_CRED_PATH = "YOUR_FIREBASE_CRED_PATH.json"
MQTT_BROKER_ADDRESS = "YOUR_BROKER_ADDRESS"
MQTT_PORT = "YOUR_MQTT_PORT"
MQTT_COMMAND_TOPIC = "YOUR_COMMAND_TOPIC"
MQTT_SENSING_TOPIC = "YOUR_SENSING_TOPIC"

class MainWindow(QMainWindow):
    mqtt_message_received = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.db = None
        self.openai_client = None
        self.mqtt_client = None
        self.plot_window = None # 차트 창을 저장할 변수

        self.mqtt_message_received.connect(self.update_sensing_text)
        self.init()

    def init(self):
        self.init_firebase()
        self.init_openai()
        self.init_mqtt()

    def init_openai(self):
        if OPENAI_API_KEY and OPENAI_API_KEY != "YOUR_OPENAI_API_KEY":
            try:
                self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
                self.ui.logText.appendPlainText("OpenAI 클라이언트 초기화 성공.")
            except Exception as e:
                QMessageBox.critical(self, "OpenAI 초기화 오류", str(e))
                self.ui.logText.appendPlainText(f"OpenAI 초기화 실패: {e}")
        else:
            QMessageBox.warning(self, "OpenAI 오류", "API 키가 설정되지 않았습니다.")
            self.ui.logText.appendPlainText("OpenAI 초기화 실패: API 키 없음.")

    def init_firebase(self):
        try:
            if not firebase_admin._apps:
                if os.path.exists(FIREBASE_CRED_PATH):
                    cred = credentials.Certificate(FIREBASE_CRED_PATH)
                    firebase_admin.initialize_app(cred)
                    self.db = firestore.client()
                    self.ui.logText.appendPlainText("Firebase 연결 성공.")
                else:
                    self.ui.logText.appendPlainText(f"Firebase 파일 없음: {FIREBASE_CRED_PATH}")
            else:
                self.db = firestore.client()
        except Exception as e:
            QMessageBox.critical(self, "Firebase 오류", str(e))
            self.ui.logText.appendPlainText(f"Firebase 연결 실패: {e}")

    def init_mqtt(self):
        try:
            self.mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            self.mqtt_client.on_connect = self.on_connect
            self.mqtt_client.on_message = self.on_message
            self.mqtt_client.connect(MQTT_BROKER_ADDRESS, MQTT_PORT, 60)
            self.mqtt_client.loop_start()
            self.ui.logText.appendPlainText(f"MQTT 연결 시도: {MQTT_BROKER_ADDRESS}")
        except Exception as e:
            QMessageBox.critical(self, "MQTT 오류", str(e))
            self.ui.logText.appendPlainText(f"MQTT 연결 실패: {e}")

    def on_connect(self, client, userdata, flags, rc, properties):
        if rc == 0:
            self.ui.logText.appendPlainText("MQTT 연결 성공.")
            client.subscribe(MQTT_SENSING_TOPIC)
        else:
            self.ui.logText.appendPlainText(f"MQTT 연결 실패: {rc}")

    def on_message(self, client, userdata, msg):
        try:
            payload = msg.payload.decode('utf-8')
            if msg.topic == MQTT_SENSING_TOPIC:
                self.mqtt_message_received.emit(payload)
        except Exception as e:
            self.ui.logText.appendPlainText(f"MQTT 메시지 오류: {e}")

    @Slot(str)
    def update_sensing_text(self, message):
        try:
            data = json.loads(message)
            formatted = json.dumps(data, indent=4, ensure_ascii=False)
            self.ui.sensingText.setPlainText(formatted)
        except json.JSONDecodeError:
            self.ui.sensingText.setPlainText(f"JSON 아님:\n{message}")
        except Exception as e:
            self.ui.sensingText.setPlainText(f"오류:\n{e}")

    def publish_command(self, cmd_string, arg_string=""):
        if not self.mqtt_client or not self.mqtt_client.is_connected():
            QMessageBox.warning(self, "MQTT 오류", "연결 안 됨")
            return
        try:
            payload = json.dumps({
                "time": time.strftime('%Y-%m-%d %H:%M:%S'),
                "cmd_string": cmd_string,
                "is_finish": "0",
                "arg_string": arg_string
            })
            result = self.mqtt_client.publish(MQTT_COMMAND_TOPIC, payload)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                self.ui.logText.appendPlainText(f"명령 발행됨: {cmd_string}")
            else:
                self.ui.logText.appendPlainText(f"발행 실패: {result.rc}")
        except Exception as e:
            self.ui.logText.appendPlainText(f"발행 오류: {e}")

    def fetch_firestore_data(self, collection="agv_log", limit=50):
        if not self.db:
            QMessageBox.warning(self, "Firestore", "연결 안 됨")
            return None
        try:
            docs_ref = self.db.collection(collection).order_by(
                "timestamp", direction=firestore.Query.DESCENDING).limit(limit)
            docs = docs_ref.stream()
            result = []
            for doc in docs:
                data = doc.to_dict()
                for k, v in data.items():
                    if hasattr(v, "isoformat"):
                        data[k] = v.isoformat()
                result.append(data)
            return json.dumps(result, indent=2, ensure_ascii=False)
        except Exception as e:
            self.ui.logText.appendPlainText(f"조회 오류: {e}")
            return None

    def query_gpt(self, prompt, firestore_data):
        """GPT에게 텍스트 분석을 요청합니다."""
        if not self.openai_client:
            return "OpenAI 미초기화"
        
        system_message = """
        당신은 AGV(무인 운반차)의 Firestore 로그 데이터를 분석하는 전문가 AI입니다.
        제공되는 JSON 데이터는 AGV의 상태 및 센서 로그 목록입니다. 
        데이터를 기반으로 사용자의 질문에 대해 명확하고 간결하며, 실행 가능한 통찰력을 제공해야 합니다.
        """
        user_message = f"""
        다음은 최근 AGV 로그 데이터입니다:
        ```json
        {firestore_data}
        ```
        이 데이터를 바탕으로 다음 질문에 답해주세요: {prompt}
        """
	print(firestore_data)
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=1024
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"GPT 오류: {e}"

    def query_gpt_for_visualization(self, prompt, firestore_data):
        """GPT에게 시각화용 Python 코드를 요청합니다."""
        if not self.openai_client:
            return "OpenAI 미초기화"
        
        try:
            df = pd.DataFrame(json.loads(firestore_data))
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
            data_description = df.head().to_string() # 데이터 샘플 제공
        except Exception as e:
            return f"데이터 변환 오류: {e}"

        system_message = """
        당신은 Python의 Matplotlib와 Pandas 라이브러리를 사용하여 데이터를 시각화하는 코드 생성 AI입니다.
        'data_df' 라는 이름의 Pandas DataFrame 변수가 이미 존재한다고 가정하고 코드를 작성하세요.
        생성된 코드는 'plt.show()' 대신 'plt.savefig(buffer, format="png", bbox_inches='tight')'를 사용하여 PNG 이미지를 버퍼에 저장해야 합니다.
        코드 블록이나 설명 없이 순수한 Python 코드만 출력해야 합니다.
        차트의 폰트가 깨지지 않도록 한글 폰트를 설정하는 코드를 포함해 주세요. 
        (예: plt.rcParams['font.family'] = 'NanumGothic', plt.rcParams['axes.unicode_minus'] = False).
        하지만, NanumGothic이 없을 수도 있으니, 기본 폰트를 사용하되, 에러가 나지 않게 하세요.
        """

        user_message = f"""
        # 데이터 샘플 (data_df 변수로 사용 가능):
        # {data_description}
        
        # 사용자의 시각화 요청: "{prompt}"
        # 아래에 Matplotlib 코드만 작성하세요:
        """

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=1024
            )
            generated_code = response.choices[0].message.content.strip()
            if generated_code.startswith("```python"):
                generated_code = generated_code[9:]
            if generated_code.endswith("```"):
                generated_code = generated_code[:-3]
            return generated_code.strip()
        except Exception as e:
            return f"GPT 오류: {e}"

    def execute_and_show_plot(self, code_string, data_df):
        """GPT가 생성한 코드를 실행하고 결과를 새 창에 표시합니다."""
        if not code_string or code_string.startswith("GPT 오류") or code_string.startswith("데이터 변환 오류"):
            self.ui.sensingText.setPlainText(f"코드 생성 실패:\n{code_string}")
            return

        self.ui.logText.appendPlainText("--- 생성된 코드 ---")
        self.ui.logText.appendPlainText(code_string)
        self.ui.logText.appendPlainText("--- 코드 실행 시작 ---")
        
        buffer = io.BytesIO()

        try:
            try:
                plt.rcParams['font.family'] = 'NanumGothic'
                plt.rcParams['axes.unicode_minus'] = False
            except Exception:
                self.ui.logText.appendPlainText("경고: 한글 폰트 설정에 실패했습니다. 차트의 한글이 깨질 수 있습니다.")

            local_vars = {'plt': plt, 'pd': pd, 'data_df': data_df, 'buffer': buffer}
            exec(code_string, {}, local_vars) # code injection에 취약함
            
            buffer.seek(0)
            pixmap = QPixmap()
            if not pixmap.loadFromData(buffer.getvalue(), "png"):
                 raise ValueError("생성된 이미지 데이터를 로드할 수 없습니다. GPT 코드를 확인하세요.")

            self.ui.sensingText.setPlainText("차트 생성 완료. 새 창을 확인하세요.")

            if self.plot_window and self.plot_window.isVisible():
                self.plot_window.close()

            self.plot_window = QWidget()
            layout = QVBoxLayout()
            label = QLabel()
            label.setPixmap(pixmap)
            layout.addWidget(label)
            self.plot_window.setLayout(layout)
            self.plot_window.setWindowTitle("GPT 생성 차트")
            self.plot_window.show()

            self.ui.logText.appendPlainText("--- 코드 실행 완료 ---")

        except Exception as e:
            self.ui.logText.appendPlainText(f"--- 코드 실행 오류 ---")
            self.ui.sensingText.setPlainText(f"코드 실행 오류:\n{e}\n\n코드:\n{code_string}")
        finally:
            plt.close('all') # 메모리 누수 방지 : 모든 Matplotlib 창 닫기 
            buffer.close()

    # 제어 명령 슬롯들
    def start(self): self.publish_command("start")
    def stop(self): self.publish_command("stop")
    def go(self): self.publish_command("go")
    def mid(self): self.publish_command("mid")
    def back(self): self.publish_command("back")
    def left(self): self.publish_command("left")
    def right(self): self.publish_command("right")

    @Slot()
    def enter(self):
        prompt = self.ui.promptText.toPlainText()
        if not prompt:
            QMessageBox.information(self, "입력 필요", "프롬프트 입력")
            return

        self.ui.logText.appendPlainText(f"프롬프트: {prompt}")
        self.ui.sensingText.setPlainText("데이터 조회 및 GPT 요청 중...")
        QApplication.processEvents()

        data_json = self.fetch_firestore_data()
        if not data_json:
            self.ui.sensingText.setPlainText("데이터 불러오기 실패")
            return

        is_visualization_request = any(keyword in prompt.lower() for keyword in ["그려줘", "차트", "시각화", "그래프"])

        if is_visualization_request:
            self.ui.sensingText.setPlainText("GPT 시각화 코드 생성 중...")
            QApplication.processEvents()
            
            try:
                data_df = pd.DataFrame(json.loads(data_json))
                if 'timestamp' in data_df.columns:
                    data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
                    data_df = data_df.sort_values('timestamp')
            except Exception as e:
                 self.ui.sensingText.setPlainText(f"데이터 변환 오류: {e}")
                 return

            code_to_run = self.query_gpt_for_visualization(prompt, data_json)
            self.execute_and_show_plot(code_to_run, data_df)
        else:
            self.ui.sensingText.setPlainText("GPT 분석 중...")
            QApplication.processEvents()
            result = self.query_gpt(prompt, data_json)
            self.ui.sensingText.setPlainText(result)
            self.ui.logText.appendPlainText("GPT 응답 완료.")

    def closeEvent(self, event):
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
        if self.plot_window and self.plot_window.isVisible():
             self.plot_window.close()
        event.accept()

if __name__ == "__main__": # Qt Designer로 UI를 만들고 pyside2-uic로 변환하여 사용해야 합니다.
    try: 
        from ui_form import Ui_MainWindow
    except ImportError:
        print("오류: ui_form.py 파일을 찾을 수 없습니다.")
        print("Qt Designer로 UI를 만들고 'pyside2-uic form.ui -o ui_form.py' 명령으로 변환하세요.")
        sys.exit(1)

    app = QApplication(sys.argv)
    
    try:
        font = QFont("NanumGothic", 10) # 한글 폰트 설정 (GUI 전체에 적용) 
        app.setFont(font)
    except Exception as e:
         print(f"경고: GUI 폰트 설정 실패 - {e}")

    widget = MainWindow()
    widget.show()
    sys.exit(app.exec_())