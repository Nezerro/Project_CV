#!/usr/bin/env python3
"""Campus Watch: локальный видеомониторинг и журнал событий для проверки человеком."""
import csv
import json
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
from PySide6.QtCore import QStandardPaths, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QComboBox, QFileDialog, QFrame, QHBoxLayout, QHeaderView,
                               QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar, QPushButton,
                               QSizePolicy, QSlider, QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)
from ultralytics import YOLO

from ai_lab import AiLabTab
from detectors import (EV_FIGHT, FallEvidence, FightModel, SensorFusion, draw_overlay)

BG, PANEL, PANEL2, LINE = "#0b1016", "#111923", "#172330", "#263545"
TEXT, MUTED, CYAN, GREEN, AMBER, RED = "#edf4fa", "#91a3b5", "#68d5e7", "#69d7a6", "#f5bd66", "#ff7585"
DEFAULT_POSE_MODEL = "yolo11n-pose.pt"

STYLE = f"""
QMainWindow, QWidget#root {{ background:{BG}; }}
QWidget {{ color:{TEXT}; font-family:'Inter','Segoe UI','Noto Sans',sans-serif; font-size:13px; }}
QLabel {{ background:transparent; border:0; }}
QFrame#card {{ background:{PANEL}; border:1px solid {LINE}; border-radius:12px; }}
QFrame#tile {{ background:{PANEL2}; border:1px solid {LINE}; border-radius:10px; }}
QLineEdit, QComboBox {{ background:{BG}; border:1px solid {LINE}; border-radius:7px; padding:8px 10px; min-height:18px; }}
QLineEdit:focus, QComboBox:focus {{ border-color:{CYAN}; }}
QComboBox::drop-down {{ border:0; width:22px; }}
QComboBox QAbstractItemView {{ background:{PANEL}; selection-background-color:#203846; border:1px solid {LINE}; }}
QSlider::groove:horizontal {{ height:5px; background:{LINE}; border-radius:2px; }}
QSlider::handle:horizontal {{ background:{CYAN}; width:15px; margin:-5px 0; border-radius:7px; }}
QProgressBar {{ background:{LINE}; border:0; border-radius:3px; min-height:6px; max-height:6px; }}
QProgressBar::chunk {{ background:{CYAN}; border-radius:3px; }}
QTabWidget::pane {{ border:0; background:{BG}; }}
QTabBar::tab {{ background:{PANEL}; color:{MUTED}; padding:9px 20px; border:1px solid {LINE}; border-bottom:0; border-top-left-radius:8px; border-top-right-radius:8px; margin-right:4px; font-weight:700; }}
QTabBar::tab:selected {{ background:{PANEL2}; color:{TEXT}; }}
QToolTip {{ background:{PANEL2}; color:{TEXT}; border:1px solid {LINE}; padding:5px; }}
QTableWidget {{ background:{BG}; border:0; selection-background-color:#203846; outline:0; }}
QTableWidget::item {{ padding:6px 8px; border-bottom:1px solid {LINE}; }}
QHeaderView::section {{ background:{PANEL2}; color:{MUTED}; border:0; padding:8px; font-size:10px; font-weight:800; }}
QScrollBar:vertical {{ background:{BG}; width:10px; }}
QScrollBar::handle:vertical {{ background:{LINE}; border-radius:5px; min-height:24px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
"""

EVENT_RED_WORDS = ("fall", "altercation", "паден", "драк")   # включая записи старого формата


def button_style(primary=False):
    return (f"QPushButton{{background:{CYAN if primary else PANEL2};color:{BG if primary else TEXT};"
            f"border:1px solid {LINE};border-radius:8px;padding:9px 15px;font-weight:700}}"
            f"QPushButton:hover{{border-color:{CYAN}}} QPushButton:disabled{{color:{MUTED}}}")


def open_capture(source):
    """Открывает камеру / файл / поток с подходящим бэкендом. None - не удалось."""
    is_stream = isinstance(source, str) and source.lower().startswith(("rtsp://", "rtsps://", "http://", "https://"))
    if isinstance(source, int):
        if sys.platform.startswith("linux"):
            backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
        elif sys.platform == "win32":
            backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]
        else:
            backends = [cv2.CAP_ANY]
    elif is_stream:
        backends = [cv2.CAP_FFMPEG, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]
    for backend in backends:
        capture = None
        try:
            if is_stream and backend == cv2.CAP_FFMPEG and hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                try:
                    capture = cv2.VideoCapture(source, backend, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                                                                  cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000])
                except TypeError:
                    capture = cv2.VideoCapture(source, backend)
            else:
                capture = cv2.VideoCapture(source, backend)
            if capture.isOpened():
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if isinstance(source, int):
                    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                return capture
        except Exception:
            pass
        if capture is not None:
            capture.release()
    return None


class CameraWorker(QThread):
    """Конвейер: этот поток читает кадры и готовит картинку для UI, а тяжёлые модели
    (поза/падение и драка) работают в собственных потоках и не тормозят видео."""
    status_ready = Signal(str, str)            # вид, текст (верхняя плашка)
    sensor_ready = Signal(str, str, str)       # fall|fight|camera, состояние, пояснение
    stats_ready = Signal(int, float)           # людей в кадре, частота анализа
    fight_score_ready = Signal(float)          # сглаженная оценка драки 0..1
    alert_ready = Signal(str, str, float, object)

    def __init__(self, source, checkpoint, sensitivity):
        super().__init__()
        self.source = source
        self.checkpoint = checkpoint or DEFAULT_POSE_MODEL
        self._sens = min(max(float(sensitivity), 0.0), 1.0)
        self._halt = threading.Event()
        self._frame_lock = threading.Lock()
        self._frame, self._frame_id = None, 0
        self._clip_lock = threading.Lock()
        self._clip = deque(maxlen=40)          # (время, кадр 224x224)
        self._ui_lock = threading.Lock()
        self._ui_frame = None
        self._alert_lock = threading.Lock()
        self._last_alert = {}
        self._dets, self._dets_time = [], 0.0
        self._people = None                    # (число, время)
        self._cap = None
        self._fall_ready, self._fight_state = False, "loading"
        self._torch_device, self._yolo_device = "cpu", "cpu"
        self._final = ("offline", "Мониторинг остановлен")

    # -- управление из UI ---------------------------------------------------- #
    def stop(self):
        self._halt.set()

    def set_sensitivity(self, value):
        self._sens = min(max(float(value), 0.0), 1.0)

    def take_ui_frame(self):
        with self._ui_lock:
            frame, self._ui_frame = self._ui_frame, None
        return frame

    # -- вспомогательное ----------------------------------------------------- #
    def _fatal(self, text, sensor=None):
        if sensor:
            self.sensor_ready.emit(sensor, "error", text[:110])
        self._final = ("error", text)
        self._halt.set()

    def _publish_status(self):
        device = self._torch_device.upper()
        if not self._fall_ready:
            self.status_ready.emit("loading", "Загрузка моделей…")
        elif self._fight_state == "ok":
            self.status_ready.emit("online", f"Работает · падения + драки · {device}")
        elif self._fight_state == "loading":
            self.status_ready.emit("loading", f"Падения работают · загрузка модели драк… · {device}")
        else:
            self.status_ready.emit("warning", f"Работает только детектор падений · модель драк недоступна · {device}")

    def _fresh_people(self, now):
        people = self._people
        return people[0] if people and now - people[1] <= 3.0 else None

    def _latest_frame(self):
        with self._frame_lock:
            return self._frame

    def _emit_alert(self, kind, detail, confidence, frame, dets):
        now = time.monotonic()
        cooldown = 4.0 + 8.0 * (1.0 - self._sens)
        with self._alert_lock:
            if now - self._last_alert.get(kind, -1e9) < cooldown:
                return
            self._last_alert[kind] = now
        shot = frame.copy()
        draw_overlay(shot, dets)
        ok, encoded = cv2.imencode(".jpg", shot, [cv2.IMWRITE_JPEG_QUALITY, 90])
        self.alert_ready.emit(kind, detail, float(confidence), bytes(encoded) if ok else b"")

    # -- главный поток конвейера --------------------------------------------- #
    def run(self):
        threads = []
        try:
            self.status_ready.emit("loading", "Открываю источник…")
            self._cap = open_capture(self.source)
            if self._cap is None:
                self.sensor_ready.emit("camera", "error", "Нет доступа к источнику")
                self._final = ("error", "Не удалось открыть источник · проверьте индекс камеры, права или адрес потока")
                return
            try:
                import torch
                cuda = torch.cuda.is_available()
            except Exception:
                cuda = False
            self._torch_device, self._yolo_device = ("cuda", 0) if cuda else ("cpu", "cpu")
            threads = [threading.Thread(target=self._pose_loop, name="pose", daemon=True),
                       threading.Thread(target=self._fight_loop, name="fight", daemon=True)]
            for thread in threads:
                thread.start()
            self._publish_status()
            self._display_loop()
        except Exception as exc:
            self._final = ("error", f"Анализ остановлен · {exc}")
        finally:
            self._halt.set()
            for thread in threads:
                thread.join(timeout=8)
            if self._cap is not None:
                self._cap.release()
            self.status_ready.emit(*self._final)

    def _display_loop(self):
        source = self.source
        is_stream = isinstance(source, str) and source.lower().startswith(("rtsp://", "rtsps://", "http://", "https://"))
        is_file = isinstance(source, str) and not is_stream
        src_fps = self._cap.get(cv2.CAP_PROP_FPS) or 0
        frame_period = 1.0 / (src_fps if 1 <= src_fps <= 120 else 25.0)
        next_tick = time.monotonic()
        stamps = deque()
        fails, last_ui, last_clip, last_report = 0, 0.0, 0.0, 0.0
        while not self._halt.is_set():
            ok, frame = self._cap.read()
            now = time.monotonic()
            if not ok or frame is None:
                if is_file:
                    self._final = ("offline", "Видео закончилось")
                    break
                fails += 1
                self.sensor_ready.emit("camera", "warn", "Нет сигнала…")
                if is_stream and fails >= 3:
                    self.status_ready.emit("warning", "Поток потерян · переподключение…")
                    self._cap.release()
                    self._halt.wait(1.0)
                    reopened = None if self._halt.is_set() else open_capture(source)
                    if reopened is not None:
                        self._cap, fails = reopened, 0
                        self._publish_status()
                elif not is_stream and fails >= 60:
                    self.sensor_ready.emit("camera", "error", "Камера не отдаёт кадры")
                    self._final = ("error", "Камера перестала отдавать кадры · проверьте подключение")
                    break
                self._halt.wait(.1)
                continue
            fails = 0
            stamps.append(now)
            while stamps and now - stamps[0] > 2.0:
                stamps.popleft()
            with self._frame_lock:
                self._frame, self._frame_id = frame, self._frame_id + 1
            if now - last_clip >= .12:                 # клип для модели драк: сразу 224x224
                last_clip = now
                small = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
                with self._clip_lock:
                    self._clip.append((now, small))
            if now - last_ui >= 1 / 30:
                last_ui = now
                view = frame.copy()
                if now - self._dets_time <= 1.0:       # не рисуем «призраков» от старого анализа
                    draw_overlay(view, self._dets)
                rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
                with self._ui_lock:
                    self._ui_frame = rgb
            if now - last_report >= 1.0:
                last_report = now
                fps = (len(stamps) - 1) / max(stamps[-1] - stamps[0], 1e-3) if len(stamps) > 1 else 0.0
                h, w = frame.shape[:2]
                self.sensor_ready.emit("camera", "ok", f"{w}×{h} · {fps:.0f} к/с")
            if is_file:                                # файл играем в реальном темпе, а не «как получится»
                next_tick += frame_period
                delay = next_tick - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -1.0:
                    next_tick = time.monotonic()

    # -- датчик падений ------------------------------------------------------ #
    def _pose_loop(self):
        try:
            self.sensor_ready.emit("fall", "loading", "Загрузка модели…")
            model = YOLO(self.checkpoint)
            sensor = FallEvidence(self._sens)
        except Exception as exc:
            self._fatal(f"Не удалось загрузить модель падений · {exc}", "fall")
            return
        self._fall_ready = True
        self.sensor_ready.emit("fall", "ok", "Анализ запущен")
        self._publish_status()
        last_id, errors, last_report = -1, 0, 0.0
        times = deque(maxlen=12)
        while not self._halt.is_set():
            with self._frame_lock:
                frame, frame_id = self._frame, self._frame_id
            if frame is None or frame_id == last_id:
                time.sleep(.01)
                continue
            last_id = frame_id
            sensor.set_sensitivity(self._sens)
            try:
                result = model.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=640,
                                     conf=sensor.det_conf, device=self._yolo_device, verbose=False)[0]
                now = time.monotonic()
                alerts, dets = sensor.inspect(result, frame.shape[0], now)
                errors = 0
            except Exception as exc:
                errors += 1
                if errors >= 5:
                    self._fatal(f"Анализ остановлен · ошибка модели падений: {exc}", "fall")
                    return
                time.sleep(.2)
                continue
            self._dets, self._dets_time, self._people = dets, now, (len(dets), now)
            for kind, detail, confidence in alerts:
                self._emit_alert(kind, detail, confidence, frame, dets)
            times.append(now)
            rate = (len(times) - 1) / max(times[-1] - times[0], 1e-3) if len(times) > 1 else 0.0
            self.stats_ready.emit(len(dets), rate)
            if now - last_report >= 1.0:
                last_report = now
                down = sum(1 for d in dets if d.state != "normal")
                if down:
                    self.sensor_ready.emit("fall", "warn", f"Человек на полу / падение: {down}")
                else:
                    self.sensor_ready.emit("fall", "ok", f"Людей в кадре: {len(dets)} · {rate:.1f} Гц")

    # -- датчик драк --------------------------------------------------------- #
    def _fight_loop(self):
        self._fight_state = "loading"
        self.sensor_ready.emit("fight", "loading", "Загрузка модели (при первом запуске скачивает веса)…")
        try:
            model = FightModel(self._torch_device)
        except Exception as exc:
            self._fight_state = "error"
            self.sensor_ready.emit("fight", "error", f"Недоступен: {str(exc)[:90]}")
            self._publish_status()
            return
        fusion = SensorFusion(self._sens)
        self._fight_state = "ok"
        self.sensor_ready.emit("fight", "ok", "Накопление кадров…")
        self._publish_status()
        last_eval, errors, warmed = 0.0, 0, False
        while not self._halt.wait(.1):
            now = time.monotonic()
            if now - last_eval < .65:
                continue
            with self._clip_lock:
                clip = [(t, f) for t, f in self._clip if now - t <= 2.6]
            if len(clip) < 8 or clip[-1][0] - clip[0][0] < 1.6:
                continue
            last_eval = now
            try:
                score = model.score([f for _, f in clip])
                errors = 0
            except Exception as exc:
                errors += 1
                if errors >= 5:
                    self._fight_state = "error"
                    self.sensor_ready.emit("fight", "error", f"Ошибка модели: {str(exc)[:90]}")
                    self._publish_status()
                    return
                continue
            if not warmed:
                warmed = True
                self.sensor_ready.emit("fight", "ok", "Анализ идёт · оценка ниже")
            fusion.set_sensitivity(self._sens)
            people = self._fresh_people(now)
            fired, smooth = fusion.update(score, people)
            self.fight_score_ready.emit(float(smooth))
            if fired:
                frame = self._latest_frame()
                if frame is not None:
                    extra = f" · людей в кадре: {people}" if people is not None else ""
                    self._emit_alert(EV_FIGHT, f"Оценка модели {score:.0%}{extra} · проверьте снимок",
                                     score, frame, list(self._dets))


class SensorTile(QFrame):
    STATES = {"ok": ("РАБОТАЕТ", GREEN), "loading": ("ЗАГРУЗКА", AMBER), "warn": ("ВНИМАНИЕ", AMBER),
              "error": ("ОШИБКА", RED), "idle": ("ОЖИДАНИЕ", MUTED)}

    def __init__(self, title, meter=False):
        super().__init__()
        self.setObjectName("tile")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(3)
        head = QLabel(title)
        head.setStyleSheet(f"font-size:10px;font-weight:800;letter-spacing:1px;color:{MUTED}")
        self.state = QLabel()
        self.detail = QLabel()
        self.detail.setWordWrap(True)
        self.detail.setStyleSheet(f"font-size:11px;color:{MUTED}")
        lay.addWidget(head)
        lay.addWidget(self.state)
        lay.addWidget(self.detail)
        self.bar = None
        if meter:
            self.bar = QProgressBar()
            self.bar.setRange(0, 100)
            self.bar.setTextVisible(False)
            lay.addWidget(self.bar)
        self.set_state("idle", "")

    def set_state(self, state, text):
        label, color = self.STATES.get(state, self.STATES["idle"])
        self.state.setText(f"●  {label}")
        self.state.setStyleSheet(f"font-size:12px;font-weight:800;color:{color}")
        self.detail.setText(text)
        self.detail.setToolTip(text)

    def set_level(self, value):
        if self.bar is not None:
            self.bar.setValue(int(round(min(max(value, 0.0), 1.0) * 100)))


class CampusWatch(QMainWindow):
    SOURCE_PLACEHOLDERS = ["Индекс камеры, например 0", "Путь к видеофайлу", "rtsp://адрес-камеры/поток"]
    PLACEHOLDER_TEXT = "<b>ВИДЕО С КАМЕРЫ</b><br><br>Выберите источник и нажмите «Начать мониторинг»"

    def __init__(self):
        super().__init__()
        self.worker = None
        self.events = []
        self.setWindowTitle("Campus Watch | Мониторинг безопасности")
        self.resize(1440, 900)
        self.setMinimumSize(1060, 680)
        self.build_ui()
        self.load_events()
        self.timer = QTimer(self)                  # кадры берём по таймеру: очередь сигналов не копится
        self.timer.timeout.connect(self.poll_frame)
        self.timer.start(33)

    # -- интерфейс ----------------------------------------------------------- #
    def build_ui(self):
        self.setStyleSheet(STYLE)
        root = QWidget()
        root.setObjectName("root")
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.ai_lab = AiLabTab(lambda: self.events)
        self.ai_lab.setObjectName("root")
        self.ai_lab.setAttribute(Qt.WA_StyledBackground, True)
        self.tabs.addTab(root, "Мониторинг")
        self.tabs.addTab(self.ai_lab, "ИИ-лаборатория")
        self.setCentralWidget(self.tabs)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(24, 20, 24, 16)
        outer.setSpacing(14)

        header = QHBoxLayout()
        brand = QVBoxLayout()
        brand.setSpacing(2)
        title = QLabel("CAMPUS WATCH")
        title.setStyleSheet("font-size:21px;font-weight:900;letter-spacing:2px")
        sub = QLabel("ЛОКАЛЬНЫЙ ВИДЕОМОНИТОРИНГ · РЕШЕНИЕ ПРИНИМАЕТ ЧЕЛОВЕК")
        sub.setStyleSheet(f"font-size:10px;letter-spacing:1.2px;color:{MUTED}")
        brand.addWidget(title)
        brand.addWidget(sub)
        header.addLayout(brand)
        header.addStretch()
        self.status_dot = QLabel()
        header.addWidget(self.status_dot)
        outer.addLayout(header)
        self.set_status("offline", "ОФЛАЙН")

        controls = QFrame()
        controls.setObjectName("card")
        col = QVBoxLayout(controls)
        col.setContentsMargins(14, 12, 14, 12)
        col.setSpacing(10)
        row1 = QHBoxLayout()
        self.source_type = QComboBox()
        self.source_type.addItems(["Камера", "Видеофайл", "Поток RTSP/HTTP"])
        self.source_type.setMinimumWidth(160)
        self.source = QLineEdit("0")
        self.source.setPlaceholderText(self.SOURCE_PLACEHOLDERS[0])
        self.source.returnPressed.connect(self.toggle_monitoring)
        self.source_browse = QPushButton("Обзор…")
        self.source_browse.setStyleSheet(button_style())
        self.source_browse.clicked.connect(self.pick_source)
        self.source_browse.setVisible(False)
        self.source_type.currentIndexChanged.connect(self.source_changed)
        row1.addWidget(self.source_type)
        row1.addWidget(self.source, 1)
        row1.addWidget(self.source_browse)
        col.addLayout(row1)

        row2 = QHBoxLayout()
        row2.setSpacing(8)
        row2.addWidget(self.caption("МОДЕЛЬ ПАДЕНИЙ"))
        self.weights = QLineEdit()
        self.weights.setPlaceholderText("Файл .pt · если пусто, используется yolo11n-pose.pt (best.pt рядом с программой подхватывается сам)")
        local = Path(__file__).resolve().with_name("best.pt")
        if local.exists():
            self.weights.setText(str(local))
        browse = QPushButton("Обзор…")
        browse.setStyleSheet(button_style())
        browse.clicked.connect(self.pick_weights)
        row2.addWidget(self.weights, 1)
        row2.addWidget(browse)
        row2.addSpacing(10)
        row2.addWidget(self.caption("ЧУВСТВИТЕЛЬНОСТЬ"))
        self.sensitivity = QSlider(Qt.Horizontal)
        self.sensitivity.setRange(0, 100)
        self.sensitivity.setValue(60)
        self.sensitivity.setFixedWidth(130)
        self.sensitivity_label = QLabel()
        self.sensitivity_label.setFixedWidth(130)
        self.sensitivity.valueChanged.connect(self.on_sensitivity)
        row2.addWidget(self.sensitivity)
        row2.addWidget(self.sensitivity_label)
        self.start_btn = QPushButton("▶  НАЧАТЬ МОНИТОРИНГ")
        self.start_btn.setStyleSheet(button_style(True))
        self.start_btn.setMinimumWidth(220)
        self.start_btn.clicked.connect(self.toggle_monitoring)
        row2.addWidget(self.start_btn)
        col.addLayout(row2)
        outer.addWidget(controls)
        self.on_sensitivity(self.sensitivity.value())

        body = QHBoxLayout()
        body.setSpacing(16)
        outer.addLayout(body, 1)

        left = QVBoxLayout()
        left.setSpacing(12)
        body.addLayout(left, 7)
        self.video = QLabel(self.PLACEHOLDER_TEXT)
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(480, 270)
        self.video.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)   # иначе окно не уменьшается
        self.video.setStyleSheet(f"background:#05080c;border:1px solid {LINE};border-radius:12px;color:{MUTED};font-size:14px")
        left.addWidget(self.video, 1)

        stats = QHBoxLayout()
        stats.setSpacing(10)
        self.people_value = self.metric_card(stats, "ЛЮДЕЙ В КАДРЕ", "—", CYAN)
        self.fps_value = self.metric_card(stats, "ЧАСТОТА АНАЛИЗА", "—", GREEN)
        self.alert_value = self.metric_card(stats, "СОБЫТИЙ В ЖУРНАЛЕ", "0", AMBER)
        left.addLayout(stats)

        sensors = QHBoxLayout()
        sensors.setSpacing(10)
        self.tiles = {"fall": SensorTile("ПАДЕНИЕ / ЧЕЛОВЕК НА ПОЛУ"),
                      "fight": SensorTile("ДРАКА", meter=True),
                      "camera": SensorTile("ВИДЕОВХОД")}
        for tile in self.tiles.values():
            sensors.addWidget(tile, 1)
        left.addLayout(sensors)

        right = QFrame()
        right.setObjectName("card")
        right.setMinimumWidth(340)
        right.setMaximumWidth(460)
        body.addWidget(right, 4)
        side = QVBoxLayout(right)
        side.setContentsMargins(14, 14, 14, 14)
        side.setSpacing(10)
        top = QHBoxLayout()
        head = QLabel("ЖУРНАЛ СОБЫТИЙ")
        head.setStyleSheet("font-weight:900;font-size:14px;letter-spacing:1px")
        top.addWidget(head)
        top.addStretch()
        for text, slot in (("Снимки", self.open_snapshots), ("CSV", self.export_csv)):
            btn = QPushButton(text)
            btn.setStyleSheet(button_style())
            btn.clicked.connect(slot)
            top.addWidget(btn)
        side.addLayout(top)
        hint = QLabel("Сначала новые · кадры хранятся только на этом компьютере")
        hint.setStyleSheet(f"font-size:10px;color:{MUTED}")
        side.addWidget(hint)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["СОБЫТИЕ", "ВРЕМЯ", "ОЦЕНКА"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setShowGrid(False)
        header_view = self.table.horizontalHeader()
        header_view.setSectionResizeMode(0, QHeaderView.Stretch)
        header_view.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header_view.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.itemSelectionChanged.connect(self.show_selected)
        self.table.cellDoubleClicked.connect(self.open_event_at)
        side.addWidget(self.table, 1)

        self.preview = QLabel("Выберите событие, чтобы увидеть кадр")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(170)
        self.preview.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.preview.setStyleSheet(f"background:#05080c;border:1px solid {LINE};border-radius:8px;color:{MUTED};font-size:11px")
        side.addWidget(self.preview)
        self.details = QLabel("")
        self.details.setWordWrap(True)
        self.details.setStyleSheet(f"color:{TEXT};font-size:11px")
        side.addWidget(self.details)
        foot = QLabel("СРАБАТЫВАНИЯ МОДЕЛЕЙ - ЭТО ПОДСКАЗКИ, А НЕ ПОДТВЕРЖДЁННЫЕ ИНЦИДЕНТЫ. Проверяйте кадр перед реакцией.")
        foot.setWordWrap(True)
        foot.setStyleSheet(f"color:{MUTED};font-size:10px;background:{PANEL2};padding:10px;border-radius:7px")
        side.addWidget(foot)

        footer = QLabel("ОБРАБОТКА НА УСТРОЙСТВЕ  •  без звука и распознавания лиц  •  без непрерывной записи")
        footer.setStyleSheet(f"color:{MUTED};font-size:10px;letter-spacing:.5px")
        outer.addWidget(footer)
        self.reset_tiles()

    @staticmethod
    def caption(text):
        label = QLabel(text)
        label.setStyleSheet(f"font-size:10px;font-weight:800;letter-spacing:.8px;color:{MUTED}")
        return label

    def metric_card(self, parent, title, value, color):
        card = QFrame()
        card.setObjectName("tile")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(2)
        label = QLabel(title)
        label.setStyleSheet(f"font-size:9px;letter-spacing:1px;font-weight:800;color:{MUTED}")
        val = QLabel(value)
        val.setStyleSheet(f"font-size:23px;font-weight:900;color:{color}")
        lay.addWidget(label)
        lay.addWidget(val)
        parent.addWidget(card, 1)
        return val

    def reset_tiles(self):
        self.tiles["fall"].set_state("idle", "Запустите мониторинг")
        self.tiles["fight"].set_state("idle", "Запустите мониторинг")
        self.tiles["fight"].set_level(0)
        self.tiles["camera"].set_state("idle", "Источник не выбран")

    # -- элементы управления ------------------------------------------------- #
    def on_sensitivity(self, value):
        if value >= 90:
            text, color = "МАКСИМУМ · ТЕСТ", RED
        elif value >= 65:
            text, color = "ВЫСОКАЯ", AMBER
        elif value >= 35:
            text, color = "СРЕДНЯЯ", GREEN
        else:
            text, color = "НИЗКАЯ", CYAN
        self.sensitivity_label.setText(f"{text} · {value}")
        self.sensitivity_label.setStyleSheet(f"color:{color};font-weight:800;font-size:10px")
        if self.worker and self.worker.isRunning():
            self.worker.set_sensitivity(value / 100.0)     # применяется на лету

    def source_changed(self, index):
        self.source.setPlaceholderText(self.SOURCE_PLACEHOLDERS[index])
        self.source.setText("0" if index == 0 else "")
        self.source_browse.setVisible(index == 1)

    def pick_source(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите видеофайл", "",
                                              "Видео (*.mp4 *.avi *.mov *.mkv *.webm *.m4v);;Все файлы (*)")
        if path:
            self.source.setText(path)

    def pick_weights(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите веса модели падений", "", "Веса YOLO (*.pt)")
        if path:
            self.weights.setText(path)

    def toggle_monitoring(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.start_btn.setEnabled(False)
            self.start_btn.setText("ОСТАНОВКА…")
            return
        try:
            text = self.source.text().strip()
            if self.source_type.currentIndex() == 0:
                try:
                    source = int(text or "0")
                except ValueError:
                    raise ValueError("Для камеры укажите её номер, например 0")
            else:
                if not text:
                    raise ValueError("Укажите путь к видео или адрес потока")
                if self.source_type.currentIndex() == 1 and not Path(text).is_file():
                    raise ValueError(f"Файл не найден:\n{text}")
                source = text
            checkpoint = self.weights.text().strip()
            if checkpoint and not Path(checkpoint).is_file():
                raise ValueError(f"Файл весов не найден:\n{checkpoint}")
        except ValueError as exc:
            QMessageBox.warning(self, "Невозможно начать", str(exc))
            return
        self.reset_tiles()
        self.worker = CameraWorker(source, checkpoint, self.sensitivity.value() / 100.0)
        self.worker.status_ready.connect(self.set_status)
        self.worker.sensor_ready.connect(self.on_sensor)
        self.worker.stats_ready.connect(self.show_stats)
        self.worker.fight_score_ready.connect(self.on_fight_score)
        self.worker.alert_ready.connect(self.save_alert)
        self.worker.finished.connect(self.worker_finished)
        self.start_btn.setText("■  ОСТАНОВИТЬ")
        self.set_status("loading", "Запуск…")
        self.worker.start()

    def worker_finished(self):
        self.start_btn.setText("▶  НАЧАТЬ МОНИТОРИНГ")
        self.start_btn.setEnabled(True)
        self.video.setPixmap(QPixmap())
        self.video.setText(self.PLACEHOLDER_TEXT)
        self.people_value.setText("—")
        self.fps_value.setText("—")
        self.reset_tiles()

    # -- обновление из потока анализа ---------------------------------------- #
    def poll_frame(self):
        if not self.worker:
            return
        rgb = self.worker.take_ui_frame()
        if rgb is None:
            return
        h, w, channels = rgb.shape
        image = QImage(rgb.data, w, h, channels * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)             # копирует данные, rgb можно отпускать
        self.video.setPixmap(pixmap.scaled(self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def on_sensor(self, key, state, text):
        tile = self.tiles.get(key)
        if tile:
            tile.set_state(state, text)

    def on_fight_score(self, value):
        self.tiles["fight"].set_level(value)

    def show_stats(self, count, rate):
        self.people_value.setText(str(count))
        self.fps_value.setText(f"{rate:.1f} Гц")

    def set_status(self, kind, text):
        color = GREEN if kind == "online" else RED if kind == "error" else AMBER if kind in ("loading", "warning") else MUTED
        shown = text if len(text) <= 80 else text[:79] + "…"
        self.status_dot.setText("●  " + shown)
        self.status_dot.setToolTip(text)
        self.status_dot.setStyleSheet(f"color:{color};font-weight:800;padding:9px 12px;background:{PANEL};"
                                      f"border:1px solid {LINE};border-radius:8px")

    # -- журнал событий ------------------------------------------------------ #
    def app_data(self):
        path = Path(QStandardPaths.writableLocation(QStandardPaths.AppDataLocation))
        path.mkdir(parents=True, exist_ok=True)
        return path

    def events_path(self):
        return self.app_data() / "events.json"

    def snapshot_dir(self):
        path = self.app_data() / "snapshots"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def load_events(self):
        try:
            self.events = json.loads(self.events_path().read_text(encoding="utf-8"))[:1000]
        except (OSError, ValueError):
            self.events = []
        for event in self.events[:200]:
            self.add_table_row(event)
        self.alert_value.setText(str(len(self.events)))

    def save_alert(self, kind, detail, confidence, jpeg):
        now = datetime.now()
        image_path = ""
        if jpeg:
            image_path = str(self.snapshot_dir() / f"{now:%Y%m%d-%H%M%S-%f}.jpg")
            try:
                Path(image_path).write_bytes(jpeg)
            except OSError:
                image_path = ""
        event = {"time": now.strftime("%Y-%m-%d %H:%M:%S"), "event": kind, "details": detail,
                 "confidence": round(float(confidence), 4), "screenshot": image_path}
        self.events.insert(0, event)
        self.events = self.events[:1000]
        try:
            tmp = self.events_path().with_suffix(".tmp")
            tmp.write_text(json.dumps(self.events, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.events_path())
        except OSError as exc:
            self.status_dot.setToolTip(f"Не удалось сохранить журнал: {exc}")
        self.add_table_row(event, prepend=True)
        self.alert_value.setText(str(len(self.events)))
        self.ai_lab.on_alert()                    # маршрут охранника к камере (A*)
        QApplication.beep()

    def add_table_row(self, event, prepend=False):
        row = 0 if prepend else self.table.rowCount()
        self.table.insertRow(row)
        self.table.setRowHeight(row, 38)
        name = QTableWidgetItem(event.get("event", "Событие"))
        critical = any(word in name.text().lower() for word in EVENT_RED_WORDS)
        name.setForeground(Qt.GlobalColor.red if critical else Qt.GlobalColor.yellow)
        score = event.get("confidence")
        self.table.setItem(row, 0, name)
        self.table.setItem(row, 1, QTableWidgetItem(event.get("time", "")[5:]))
        self.table.setItem(row, 2, QTableWidgetItem(f"{float(score):.0%}" if score is not None else "—"))
        while self.table.rowCount() > 200:
            self.table.removeRow(self.table.rowCount() - 1)

    def show_selected(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows or rows[0].row() >= len(self.events):
            return
        event = self.events[rows[0].row()]
        self.details.setText(f"{event.get('time', '')} · {event.get('details', '')}")
        path = event.get("screenshot", "")
        pixmap = QPixmap(path) if path and Path(path).exists() else QPixmap()
        if pixmap.isNull():
            self.preview.setPixmap(QPixmap())
            self.preview.setText("Снимок недоступен")
        else:
            self.preview.setPixmap(pixmap.scaled(self.preview.width(), 220, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def open_event_at(self, row, _column):
        if row < len(self.events):
            self.open_snapshot(self.events[row].get("screenshot", ""))

    def open_snapshot(self, path):
        if path and Path(path).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        else:
            QMessageBox.information(self, "Снимок недоступен", "Для этого события нет сохранённого снимка.")

    def open_snapshots(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.snapshot_dir())))

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Экспорт журнала", "campus-watch-events.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as handle:   # utf-8-sig: Excel читает кириллицу
                writer = csv.DictWriter(handle, fieldnames=["time", "event", "confidence", "details", "screenshot"])
                writer.writeheader()
                writer.writerows(reversed(self.events))
        except OSError as exc:
            QMessageBox.warning(self, "Экспорт не удался", str(exc))

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(10000)
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("campus-watch")
    app.setOrganizationName("CampusWatch")
    app.setStyle("Fusion")
    window = CampusWatch()
    window.show()
    sys.exit(app.exec())
