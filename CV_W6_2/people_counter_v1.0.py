######################################################################################################
# people_counter_v1.0.py : 2023.8.15.
# Author : C.W. Lim
#
# GUI implementation done
#   (1) configure detect line postion
#   (2) GUI componants implemented
#   (3) Count number display
#   (4) log display
#   (5) log to be saved into 'mylog.txt'
#
# ByteTrack + Yolov8 integration done
#   (1) porting ByteTracker + Yolov8
#   (2) display detect line
#   (3) display in_cnt
# To do 
#   (1) tuning the performance : speed, tracking
#   (2) custom data training

##### GUI
import cv2
import threading
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import time
import logging

##### ByteTracker + Yolov8
import os
from IPython import display
import ultralytics
from ultralytics import YOLO
import yolox

from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
import supervision
from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.notebook.utils import show_frame_in_notebook
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator

from typing import List
import numpy as np

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Configure the logger
log_format = "%(levelname)s %(asctime)s - %(message)s"
logging.basicConfig(filename="my_log.txt", filemode="w", format=log_format, level=logging.DEBUG)

# Create a logger object
logger = logging.getLogger()

# Log messages
logger.debug("Start people counter")
logger.info("This is an info message")
logger.warning("This is a warning message")
logger.error("This is an error message")
logger.critical("This is a critical message")

##### Define Environment Configuration Variables
ENV_Show_Log = False
# ENV_Show_Log = True

# Define Color
blue_color = (255, 0, 0)
green_color = (0, 255, 0)
red_color = (0, 0, 255)
white_color = (255, 255, 255)
black_color = (0, 0, 0)

# Define camera resolution
DCamWidth = 640
DCamHeight = 480

HOME = os.getcwd()
DATA_HOME = f'{HOME}//data'
if ENV_Show_Log:
    logger.debug(f"HOME: {HOME}")
    logger.debug(f"DATA_HOME:{DATA_HOME}")

display.clear_output()
ultralytics.checks()

os.chdir(f'{HOME}/ByteTrack')
current_dir = os.getcwd()
if ENV_Show_Log:
    logger.debug(f"Current working folder:{current_dir}")
    logger.debug(f"yolox.__version__:{yolox.__version__}")
    logger.debug(f"supervision.__version__:{supervision.__version__}")

@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))


# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)


# matches our bounding boxes with predictions
def match_detections_with_tracks(
    detections: Detections, 
    tracks: List[STrack]
) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)
    
    tracker_ids = [None] * len(detections)
    
    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids

# settings
# MODEL = "C:/Users/User2/source/pcounter/20230829-초기학습/yolov8x.pt"
MODEL = "C:/Users/chaew/source/pcounter/20230829-초기학습/yolov8x.pt"
# MODEL = "yolov8s.pt"
# MODEL = "C:/Users/User2/source/pcounter/20230831-학습용데이터증강/runs/detect/train/weights/best.pt"
# MODEL = "yolov8n.pt"

model = YOLO(MODEL)
model.info()
model.fuse()

# dict maping class_id to class_name
CLASS_NAMES_DICT = model.model.names
# class_ids of interest - person, car, motorcycle, bus, truck, horses, sports ball, cup, cell phone
# CLASS_ID = [0, 2, 3, 5, 7, 17, 32, 41, 67]
# CLASS_ID = [41] # cup
CLASS_ID = [0] # person
# CLASS_ID = [67] # cell phone

if ENV_Show_Log:
    logger.debug(f"{CLASS_NAMES_DICT}")

TARGET_VIDEO_PATH = f"{DATA_HOME}/outpy_result2.mp4"

# camera
import cv2
import numpy as np
import screeninfo

# screeen
screen_id = 0
#is_color = False

# get the size of the screen
screen = screeninfo.get_monitors()[screen_id]
if ENV_Show_Log:
    logger.debug(f"scr w,h: {screen.width}, {screen.height}")

class PeopleCount(QWidget):

    def __init__(self) :
        super().__init__()
        self.setWindowTitle('입장 인원 카운터 v0.8')	# 윈도우 이름과 위치 지정
        # self.setGeometry(200,200,500,100)
        self.showFullScreen()


        self.CamID = 0 # 캠 1개만 있을 경우 - 내장 캠, 캠 2개 있을 경우 - 외장 캠
        # self.CamID = 1 # 캠 2개 있을 경우 - 내장 캠

        self.det_line_st = QPoint(0,0)        
        self.det_line_ed = QPoint(0,0)

        self.LINE_START = Point(int(self.det_line_st.x()), int(self.det_line_st.y()))
        self.LINE_END = Point(int(self.det_line_ed.x()), int(self.det_line_ed.y()))
        self.L_S = self.LINE_START.as_xy_int_tuple()
        self.L_E = self.LINE_END.as_xy_int_tuple()


        self.cnt_per_unit_time = 0
        self.in_cnt = 0
        self.in_cnt_pre = 0

        self.unitsec = 1000
        self.unitmin = 60000
        self.unithour = 3600000

        # self.n_sec = 90
        self.n_sec = 3600
        self.n_sec_timer_duration = self.n_sec*self.unitsec # n sec
        self.n_min_timer_duration = self.n_sec*self.unitmin # n min
        self.n_hour_timer_duration = self.n_sec*self.unithour # n hour

        self.run_counting = False
        self.configured = False
        self.running = False

        self.lb_title = QPushButton("마이크로디그리 전공박람회 방문을 환영합니다.")
        self.lb_title.setStyleSheet("font-family: Arial; font-size: 70px; color: black;")        
        # self.lb_title.setAlignment(Qt.AlignCenter)
        self.lb_title.setCheckable(True)
        self.lb_title.toggle()

        self.lcd = QLCDNumber(self)

        self.lb_pic = QLabel()
        self.lb_pic.setScaledContents(True)
        # self.lb_pic.resize(width, height)


        self.text_edit = QTextEdit(self)
        self.text_edit.setAlignment(Qt.AlignBottom)
        self.text_edit.setReadOnly(True)
        self.clear_button = QPushButton("Clear log", self)
        self.log_lines = []  # 로그 라인을 저장할 리스트

        self.lb_info1 = QLabel()
        self.lb_info2 = QLabel()
        # self.lb_info.setText(f'w = {width}, h = {height}')

        # self.lb_cnt_title = QLabel("PEOPLE ENTERED:", self)
        self.lb_cnt_title = QLabel("박람회 참여자:", self)
        self.lb_cnt_title.setStyleSheet("font-family: Arial; font-size: 60px; color: blue;")
        self.lb_cnt_title.setAlignment(Qt.AlignHCenter|Qt.AlignBottom)

        self.btn_config = QPushButton("Config detection area")
        # btn_camera_on = QPushButton("Camera on")
        self.btn_start = QPushButton("Run count")
        self.btn_stop = QPushButton("Exit program")
        
        vbox1 = QVBoxLayout()
        vbox1.addWidget(self.text_edit, stretch = 10)
        vbox1.addWidget(self.clear_button, stretch = 2)
        vbox1.addWidget(self.lb_info1, stretch = 1)
        vbox1.addWidget(self.lb_info2, stretch = 1)
        vbox1.addWidget(self.lb_cnt_title, stretch = 4)
        vbox1.addWidget(self.lcd, stretch = 8)
        vbox1.stretch(1)

        hbox1 = QHBoxLayout()
        hbox1.addWidget(self.lb_pic, stretch = 2)
        hbox1.addLayout(vbox1, stretch=1)

        hbox2 = QHBoxLayout()
        hbox2.addWidget(self.btn_config)
        # hbox2.addWidget(btn_camera_on)
        hbox2.addWidget(self.btn_start)
        hbox2.addWidget(self.btn_stop)

        vbox2 = QVBoxLayout()
        vbox2.addWidget(self.lb_title)
        vbox2.addLayout(hbox1)
        vbox2.addLayout(hbox2)
        self.setLayout(vbox2)

        self.timer = QBasicTimer()
        self.step = 0

        # btn_camera_on.clicked.connect(self.camera_on)
        self.clear_button.clicked.connect(self.clear_log)
        self.btn_config.clicked.connect(self.config_det_area)
        self.btn_start.clicked.connect(self.start_counting)
        self.btn_stop.clicked.connect(self.stop)
        app.aboutToQuit.connect(self.onExit)


    # def append_log(self, log_text):
    #     self.log_lines.append(log_text)  # 새로운 로그 라인 추가
    #     self.text_edit.append(log_text)   # 로그 라인을 TextEdit에 추가
        
    #     max_log_lines = 100  # 최대 보존 로그 라인 수
    #     if len(self.log_lines) > max_log_lines:
    #         self.log_lines.pop(0)  # 가장 오래된 로그 라인 삭제
    #         self.text_edit.moveCursor(QTextCursor.Start)
    #         self.text_edit.moveCursor(QTextCursor.Down, QTextCursor.KeepAnchor)
    #         self.text_edit.moveCursor(QTextCursor.EndOfLine, QTextCursor.KeepAnchor)
    #         self.text_edit.textCursor().removeSelectedText()
        
    #     self.text_edit.moveCursor(QTextCursor.End)  # 로그가 추가될 때마다 스크롤

    def timerEvent(self, e):
        self.timer.stop()
        self.cnt_per_unit_time += max(0, self.in_cnt - self.in_cnt_pre)
        self.in_cnt_pre = self.in_cnt

        datetime = QDateTime.currentDateTime()
        time_string = datetime.toString('dd.MM.yyyy, hh:mm:ss')
        n_sec = self.n_sec_timer_duration/self.unitsec

        if n_sec > 3600:
            self.print_log(f'{self.cnt_per_unit_time} persons / {n_sec*self.unitsec/self.unithour} hour entererd @ {time_string}')
        elif n_sec > 60:
            self.print_log(f'{self.cnt_per_unit_time} persons / {n_sec*self.unitsec/self.unitmin} min entererd @ {time_string}')
        else:
            self.print_log(f'{self.cnt_per_unit_time} persons / {n_sec} sec entererd @ {time_string}')

        self.cnt_per_unit_time = 0
        self.timer.start(self.n_sec_timer_duration, self)
        if self.step >= n_sec:
            self.step = 0
            return

        self.step = self.step + 1

    def start_timedlog(self):
        self.step = 0
        datetime = QDateTime.currentDateTime()
        time_string = datetime.toString('dd.MM.yyyy, hh:mm:ss')

        n_sec = self.n_sec_timer_duration/self.unitsec
        self.timer.start(self.n_sec_timer_duration, self)

        if n_sec > 3600:
            self.print_log(f'Start {n_sec*self.unitsec/self.unithour} hour timer @ {time_string}')
        elif n_sec > 60:
            self.print_log(f'Start {n_sec*self.unitsec/self.unitmin} min timer @ {time_string}')
        else:
            self.print_log(f'Start {n_sec} sec timer @ {time_string}')

    def append_log(self, log_text):
        self.log_lines.append(log_text)
        self.text_edit.append(log_text)
        
        max_log_lines = 40
        if len(self.log_lines) > max_log_lines:
            self.log_lines.pop(0)
        
        self.text_edit.moveCursor(QTextCursor.End)  # Move cursor to the end
        self.text_edit.ensureCursorVisible()  # Autoscroll to the bottom

    # def append_log(self, log_text):
    #     self.log_lines.append(log_text)
    #     self.text_edit.append(log_text)
        
    #     max_log_lines = 40
    #     if len(self.log_lines) > max_log_lines:
    #         self.log_lines.pop(0)
        
    #     # Instead of explicitly manipulating the cursor, use invokeMethod
    #     # to ensure GUI updates are done in the main thread
    #     QMetaObject.invokeMethod(self.text_edit, 'moveCursor', Qt.QueuedConnection, Q_ARG(QTextCursor.MoveOperation, QTextCursor.End))
    #     QMetaObject.invokeMethod(self.text_edit, 'ensureCursorVisible', Qt.QueuedConnection)

    def print_log(self, log):
        # print out to edit box
        self.append_log(log)

        # Log messages
        logger.debug(log)

    def clear_log(self):
        self.text_edit.setPlainText("")  # 로그 창을 지웁니다.
        self.log_lines.clear()  # 로그 라인 리스트도 초기화
    
    def config_det_area(self):
        if self.configured == False:
            # Create a VideoCapture object
            self.cap = cv2.VideoCapture(self.CamID)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, DCamWidth)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DCamHeight)

            # Check if camera opened successfully
            if (self.cap.isOpened() == False):
                logger.debug("Unable to read camera feed")

            # Default resolutions of the frame are obtained.The default resolutions are system dependent.
            # We convert the resolutions from float to integer.
            width = int(self.cap.get(3))
            height = int(self.cap.get(4))
            if ENV_Show_Log:
                logger.debug(f"frame size: ({width}, {height})")

            self.print_log("Configuring..")
            for i in range(10):
                ret, self.img = self.cap.read()
                cv2.imshow('Line drawing',self.img)
                cv2.waitKey(1)
                # self.print_log("imshow(self.img)")
            cv2.setMouseCallback('Line drawing',self.linedrawing) 

            # self.cap.release()

            # self.camera_on()
        
    def linedrawing(self,event,x,y,flags,param):
        if event==cv2.EVENT_LBUTTONDOWN:   
            self.det_line_st.setX(x)
            self.det_line_st.setY(y)
            # cv2.circle(self.img, (int(x),int(y)), 10, (0,0,255), 4, lineType=-1)
            cv2.circle(self.img, (x,y), 10, (0,0,255), 4, lineType=-1)
            cv2.imshow('Line drawing',self.img)
            cv2.waitKey(1)

        elif event==cv2.EVENT_MOUSEMOVE and flags==cv2.EVENT_FLAG_LBUTTON:
            self.det_line_ed.setX(x)
            self.det_line_ed.setY(y)
            self.img_disp = self.img.copy()
            cv2.line(self.img_disp, (int(self.det_line_st.x()),int(self.det_line_st.y())), (x,y), (80, 80, 80), thickness=4)            
            cv2.imshow('Line drawing',self.img_disp)
            cv2.waitKey(1)

        elif event==cv2.EVENT_LBUTTONUP:
            cv2.circle(self.img, (int(self.det_line_st.x()),int(self.det_line_st.y())), 10, (0,0,255), 4, lineType=-1)
            cv2.line(self.img, (int(self.det_line_st.x()),int(self.det_line_st.y())), (int(self.det_line_ed.x()),int(self.det_line_ed.y())), (0,0,255), thickness=4)
            cv2.circle(self.img, (x,y), 10, (0,0,255), 4, lineType=-1)
            cv2.imshow('Line drawing',self.img)
            cv2.waitKey(1)
            self.det_line_ed.setX(x)
            self.det_line_ed.setY(y)

            self.cap.release()

            self.camera_on()
            cv2.destroyWindow('Line drawing')

            self.LINE_START = Point(int(self.det_line_st.x()), int(self.det_line_st.y()))
            self.LINE_END = Point(int(self.det_line_ed.x()), int(self.det_line_ed.y()))
            self.L_S = self.LINE_START.as_xy_int_tuple()
            self.L_E = self.LINE_END.as_xy_int_tuple()

            # create BYTETracker instance
            self.byte_tracker = BYTETracker(BYTETrackerArgs())

            # create LineCounter instance
            self.line_counter = LineCounter(start=self.LINE_START, end=self.LINE_END)

            # create instance of BoxAnnotator and LineCounterAnnotator
            self.box_annotator = BoxAnnotator(color=ColorPalette(), thickness=1, text_scale=0.5, text_thickness=1)
            # self.line_annotator = LineCounterAnnotator(thickness=2, text_thickness=2, text_scale=0.5)


            self.print_log(f"Configuring done! ({self.det_line_st.x()}, {self.det_line_st.y()}) - ({self.det_line_ed.x()}, {self.det_line_ed.y()})")
            self.configured=True


            # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
            # out = cv2.VideoWriter(TARGET_VIDEO_PATH, cv2.VideoWriter_fourcc(*'FMP4'), 30, (width,height)) # OpenCV: FFMPEG: tag 0x34504d46/'FMP4' is not supported with codec id 12 and format 'mp4 / MP4 (MPEG-4 Part 14)'
            # self.out = cv2.VideoWriter(TARGET_VIDEO_PATH, cv2.VideoWriter_fourcc(*'MP4V'), 30, (width,height))
            self.out = cv2.VideoWriter(TARGET_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), 30, (DCamWidth,DCamHeight)) # OpenCV: FFMPEG: tag 0x5634504d/'MP4V' is not supported with codec id 12 and format 'mp4 / MP4 (MPEG-4 Part 14)'
                                                                                                               # OpenCV: FFMPEG: fallback to use tag 0x7634706d/'mp4v'

            # img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

            # qImg = QImage(img_rgb.data, self.img.shape(1), self.img.shape(0), QImage.Format_RGB888)
            # pixmap = QPixmap.fromImage(qImg)
            # self.lb_pic.setPixmap(pixmap)

    def camera_on(self):
        self.run_counting = False
        # self.congfigured = False

        if self.running == False:
            th = threading.Thread(target=self.run)
            th.start()
            self.running = True

        self.print_log("Camera on..")
        
    def run(self):
        self.cap = cv2.VideoCapture(self.CamID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, DCamWidth)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DCamHeight)
        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.lb_pic.resize(int(width), int(height))

        self.lb_info1.setText(f'카메라 해상도: ({int(width)}, {int(height)})')
        self.lb_info1.setAlignment(Qt.AlignCenter)
        self.lb_info1.setStyleSheet("font-family: Arial; font-size: 20px; color: black;")     

        datetime = QDateTime.currentDateTime()
        time_string = datetime.toString('yyyy.MM.dd, hh:mm:ss')
        self.lb_info2.setText(f'카운팅 시작시각: {time_string}')
        self.lb_info2.setAlignment(Qt.AlignCenter)
        self.lb_info2.setStyleSheet("font-family: Arial; font-size: 20px; color: black;") 

        frame_id = 0
        pre_time = time.time()
        while self.running:
            ret, self.img = self.cap.read()
            if ret:
                # print(f'{self.img.shape}')
                # if self.configured == False:
                #     # cv2.imshow('Line drawing',self.img)
                #     # cv2.waitKey(1)
                #     # print("imshow(self.img)")

                # else:
                if self.run_counting:
                    # detect
                    results = model(self.img)
                    detections = Detections(
                        xyxy=results[0].boxes.xyxy.cpu().numpy(),
                        confidence=results[0].boxes.conf.cpu().numpy(),
                        class_id=results[0].boxes.cls.cpu().numpy().astype(int)
                    )

                    # filtering out detections with unwanted classes
                    mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
                    detections.filter(mask=mask, inplace=True)

                    # tracking detections
                    tracks = self.byte_tracker.update(
                        output_results=detections2boxes(detections=detections),
                        img_info=self.img.shape,
                        img_size=self.img.shape
                    )
                    tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
                    detections.tracker_id = np.array(tracker_id)

                    # filtering out detections without trackers
                    mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
                    detections.filter(mask=mask, inplace=True)

                    # format custom labels
                    labels = [
                        f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                        for _, confidence, class_id, tracker_id
                        in detections
                    ]

                    # updating line counter
                    self.line_counter.update(detections=detections)
                    logger.debug(f"line_counter : {self.line_counter.in_count}, {self.line_counter.out_count}")
                    self.in_cnt = self.line_counter.in_count

                    # annotate and display frame
                    self.img = self.box_annotator.annotate(frame=self.img, detections=detections, labels=labels)
                    # self.line_annotator.annotate(frame=self.img, line_counter=self.line_counter)
                    # cv2.line(frame, L_S, L_E, color=red_color,thickness=4)
                    # cv2.putText(frame, f'In  : {line_counter.in_count}', (30, 80), cv2.FONT_HERSHEY_DUPLEX, fontScale=1.5, color=blue_color, thickness=4, lineType=cv2.LINE_AA)
                    # cv2.putText(frame, f'Out: {line_counter.out_count}', (30, 130), cv2.FONT_HERSHEY_DUPLEX, fontScale=1.5, color=blue_color, thickness=4, lineType=cv2.LINE_AA)

                    # Write the frame into the file
                    self.out.write(self.img) 

                    # self.in_cnt = self.count_people(self.img, frame_id)
                    # self.print_log(f'number of persons = {self.in_cnt}')
                    logger.debug(f'number of persons = {self.in_cnt} @ {frame_id}')
                    self.lcd.display(self.in_cnt)
                else:
                    logger.debug('no count process')


                # cv2.line(self.img, (int(self.det_line_st.x()), int(self.det_line_st.y())), (int(self.det_line_ed.x()), int(self.det_line_ed.y())), (0,0,255), thickness=4, lineType=-1)
                # cv2.circle(self.img, (int(self.det_line_st.x()), int(self.det_line_st.y())), 10, (0,0,0), 4, lineType=cv2.LINE_AA)
                # cv2.circle(self.img,  (int(self.det_line_ed.x()), int(self.det_line_ed.y())), 10, (0,0,0), 4, lineType=cv2.LINE_AA)

                img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

                qImg = QImage(img_rgb.data, int(width), int(height), QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qImg)
                self.lb_pic.setPixmap(pixmap)
                # print("lb_pic.setPixmap(pixmap)")

                # print(f'frame time = {time.time() - pre_time}sec')
                frame_id += 1
                pre_time = time.time()
            else:
                QMessageBox.about(win, "Error", "Cannot read frame.")
                self.print_log("Cannot read frame.")
                break

        self.cap.release()
        self.out.release()
        self.print_log("Thread end.")
        self.stop()
  
    # def count_people(self, frame, i):
    #     st = time.time()        
    #     if self.configured:
    #         # self.print_log(f"processing people count {i}")           
    #         print(f"processing people count {i}")           
    #         self.lcd.display(i)
    #         time.sleep(0.025)
    #     else:
    #         self.print_log('Configuration should be done first!')

    #     # self.print_log(f'processing time = {time.time() - st}')
    #     print(f'processing time = {time.time() - st}')
    #     return i    

    def start_counting(self):
        if self.configured:
            self.run_counting = True
            self.print_log("Counting..")
            self.start_timedlog()
        else:
            self.print_log('Configuration should be done first!')

    def stop(self):
        self.running = False
        self.print_log("Stopped")
        self.close()

        
    def onExit(self):
        self.print_log("Exit")
        self.stop()

app = QApplication(sys.argv)
win = PeopleCount()
win.show()
sys.exit(app.exec_())