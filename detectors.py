"""Campus Watch: логика датчиков (падение, драка) и отрисовка разметки.

Все срабатывания - подсказки для оператора, а не диагноз: решение всегда за человеком.
Модуль не зависит от Qt и ultralytics, поэтому его можно тестировать отдельно.
"""
from __future__ import annotations

import contextlib
import math
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

EV_FALL = "Возможное падение"
EV_LYING = "Человек лежит"
EV_FIGHT = "Возможная драка"


def _clamp01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


# --------------------------------------------------------------------------- #
# Данные и отрисовка
# --------------------------------------------------------------------------- #
@dataclass
class Detection:
    track: Optional[int]
    label: str
    conf: float
    box: tuple
    kind: str = "person"      # person | fall
    state: str = "normal"     # normal | lying | fall


# BGR-версии цветов интерфейса: cyan, amber, red
_COLORS = {"normal": (231, 213, 104), "lying": (102, 189, 245), "fall": (133, 117, 255)}


def draw_overlay(frame, detections):
    """Рисует рамки на кадре (на месте). Текст только ASCII: шрифты cv2 не умеют кириллицу."""
    height = frame.shape[0]
    scale = max(0.45, height / 1200)
    thick = max(1, int(round(height / 480)))
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.box)
        color = _COLORS.get(det.state, _COLORS["normal"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick + (1 if det.state != "normal" else 0))
        tag = {"lying": "LYING", "fall": "FALL?"}.get(det.state, "")
        parts = [f"ID{det.track}" if det.track is not None else "", tag, f"{det.conf:.0%}"]
        text = " ".join(p for p in parts if p)
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        top = max(y1 - th - base - 4, 0)
        cv2.rectangle(frame, (x1, top), (x1 + tw + 6, top + th + base + 4), color, -1)
        cv2.putText(frame, text, (x1 + 3, top + th + 1), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (22, 16, 11), 1, cv2.LINE_AA)
    return frame


# --------------------------------------------------------------------------- #
# Драка: видеомодель X3D-M
# --------------------------------------------------------------------------- #
def _locate_fight_weights() -> str:
    """Сначала локальные веса (работает офлайн), затем скачивание с Hugging Face."""
    candidates = [os.environ.get("CAMPUS_FIGHT_WEIGHTS"),
                  Path(__file__).resolve().with_name("final_x3d_realtime.pt")]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    from huggingface_hub import hf_hub_download
    return hf_hub_download("visionlab-ai/school-violence-detection-models", "final/final_x3d_realtime.pt")


class FightModel:
    """X3D-M: 16 кадров, равномерная выборка, 224x224, нормализация как у автора модели."""

    def __init__(self, device="cpu"):
        import torch
        from pytorchvideo.models.hub import x3d_m

        self.torch = torch
        self.device = device
        path = _locate_fight_weights()

        class ViolenceX3D(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = x3d_m(pretrained=False)
                features = self.backbone.blocks[-1].proj.in_features
                self.backbone.blocks[-1].proj = torch.nn.Sequential(
                    torch.nn.Dropout(.5), torch.nn.Linear(features, 2))

            def forward(self, value):
                return self.backbone(value)

        model = ViolenceX3D()
        numpy_core = getattr(np, "_core", None) or getattr(np, "core")
        safe = [numpy_core.multiarray.scalar, np.dtype]
        if hasattr(np, "dtypes") and hasattr(np.dtypes, "Float64DType"):
            safe.append(np.dtypes.Float64DType)
        # weights_only=True оставляем всегда: не исполняем произвольный pickle из чужого файла.
        guard = (torch.serialization.safe_globals(safe)
                 if hasattr(torch.serialization, "safe_globals") else contextlib.nullcontext())
        with guard:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
        if state and all(key.startswith("module.") for key in state):
            state = {key[7:]: value for key, value in state.items()}
        model.load_state_dict(state, strict=True)
        self.model = model.eval().to(device)

    def score(self, bgr_frames) -> float:
        """Вероятность класса «драка» для списка BGR-кадров (любой длины >= 2)."""
        if len(bgr_frames) < 2:
            raise ValueError("нужно как минимум 2 кадра")
        indices = np.linspace(0, len(bgr_frames) - 1, 16).astype(int)
        rgb = []
        for i in indices:
            frame = bgr_frames[i]
            if frame.shape[0] != 224 or frame.shape[1] != 224:
                frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
            rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        value = np.stack(rgb).astype(np.float32) / 255.0
        tensor = self.torch.from_numpy(value).permute(3, 0, 1, 2).contiguous()
        tensor = ((tensor - .45) / .225).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            return float(self.torch.softmax(self.model(tensor), dim=1)[0, 1].item())


class SensorFusion:
    """Сглаживание оценок драки во времени.

    Порог зависит от чувствительности, но не от входа модели. Один эпизод даёт одно
    срабатывание: повторно датчик «вооружается», только когда оценка заметно упала.
    """

    def __init__(self, sensitivity=0.6):
        self.scores = deque(maxlen=3)
        self.smooth: Optional[float] = None
        self.active = False
        self.set_sensitivity(sensitivity)

    def set_sensitivity(self, sensitivity):
        s = self.sensitivity = _clamp01(sensitivity)
        self.threshold = .90 - .35 * s              # 0.90 ... 0.55
        self.min_people = 2 if s < .95 else 0       # драка - это минимум двое в кадре

    def update(self, score: float, people: Optional[int] = None):
        """Возвращает (сработал_ли_датчик, сглаженная_оценка).
        people=None - число людей неизвестно, проверку по людям пропускаем."""
        self.scores.append(float(score))
        self.smooth = score if self.smooth is None else .5 * self.smooth + .5 * score
        if self.active:
            if self.smooth < self.threshold - .20:
                self.active = False
            return False, self.smooth
        need = 1 if self.sensitivity >= .95 else 2
        hits = sum(x >= self.threshold for x in self.scores)
        if hits >= need and self.smooth >= self.threshold - .10:
            if people is not None and people < self.min_people:
                return False, self.smooth
            self.active = True
            return True, self.smooth
        return False, self.smooth


# --------------------------------------------------------------------------- #
# Падение: метки модели + временная логика по позе
# --------------------------------------------------------------------------- #
_NEGATIONS = ("not_", "not ", "no_", "no ", "non", "normal", "nofall", "without")
_PERSON_WORDS = ("person", "people", "human", "pedestrian", "stand", "walk", "sit",
                 "adult", "child", "student")


def is_fall_label(label: str) -> bool:
    low = label.lower().replace("-", "_")
    if any(word in low for word in _NEGATIONS):
        return False
    return any(word in low for word in ("fall", "lying", "collapse"))


def is_person_label(label: str, has_keypoints: bool) -> bool:
    return has_keypoints or any(word in label.lower() for word in _PERSON_WORDS)


class _State:
    __slots__ = ("hist", "last_seen", "lying_since", "unlying_since", "transition",
                 "alerted", "model_since", "model_last", "last_model_alert")

    def __init__(self, now):
        self.hist = deque()            # (t, cx, cy, w, h, aspect, lying)
        self.last_seen = now
        self.lying_since = None
        self.unlying_since = None
        self.transition = False        # был ли человек на ногах прямо перед «лёжа»
        self.alerted = False
        self.model_since = None
        self.model_last = 0.0
        self.last_model_alert = -1e9


class FallEvidence:
    """Падение = человек был на ногах, резко «сложился» и остаётся лёжа дольше hold секунд.

    Без предшествующего падения (человек вошёл в кадр уже лёжа, ID сменился и т.п.)
    выдаётся менее срочное «Человек лежит» - после long_hold секунд.
    Всё считается во времени (секунды), а не в кадрах: частота анализа плавает.
    """
    HISTORY_S = 6.0
    LOST_S = 2.0
    RECOVER_S = 1.0
    WINDOW_S = 3.5

    def __init__(self, sensitivity=0.6):
        self.states: dict = {}
        self.set_sensitivity(sensitivity)

    def set_sensitivity(self, sensitivity):
        s = self.sensitivity = _clamp01(sensitivity)
        self.hold = 1.8 - 1.0 * s            # с лёжа после падения: 1.8 ... 0.8
        self.long_hold = 10.0 - 5.0 * s      # с лёжа без падения: 10 ... 5
        self.tilt_thr = 62.0 - 14.0 * s      # угол торса от вертикали, градусы
        self.aspect_tilt = .95 - .25 * s     # w/h рамки при известном наклоне
        self.aspect_only = 1.5 - .3 * s      # w/h рамки без ключевых точек
        self.shrink = .65 + .15 * s          # рамка сжалась до этой доли высоты = падение
        self.model_conf = .55 - .30 * s      # порог метки «fall» кастомной модели
        self.model_hold = .8 - .5 * s        # сколько секунд метка должна держаться

    @property
    def det_conf(self) -> float:
        """Порог уверенности детектора (меньше model_conf, чтобы метки не терялись)."""
        return .45 - .25 * self.sensitivity

    # -- геометрия ---------------------------------------------------------- #
    @staticmethod
    def _tilt(points) -> Optional[float]:
        """Угол торса от вертикали: 0 - стоит, 90 - лежит. None - ключевые точки ненадёжны."""
        try:
            if len(points) <= 12:
                return None
            idx = (5, 6, 11, 12)
            if min(float(points[j][2]) for j in idx) < .4:
                return None
            shoulder = (np.asarray(points[5][:2]) + np.asarray(points[6][:2])) / 2
            hip = (np.asarray(points[11][:2]) + np.asarray(points[12][:2])) / 2
            dx, dy = abs(float(shoulder[0] - hip[0])), abs(float(shoulder[1] - hip[1]))
            if dx + dy < 4:
                return None
            return math.degrees(math.atan2(dx, dy))
        except (IndexError, TypeError, ValueError):
            return None

    def _is_lying(self, aspect: float, tilt: Optional[float]) -> bool:
        if tilt is not None:
            return tilt >= self.tilt_thr and aspect >= self.aspect_tilt
        return aspect >= self.aspect_only

    def _had_transition(self, st: _State, now: float, height: float, aspect: float) -> bool:
        upright = [e for e in st.hist if now - e[0] <= self.WINDOW_S and not e[6]]
        if not upright:
            return False
        ref_h = max(e[4] for e in upright)
        ref_aspect = min(e[5] for e in upright)
        return height <= ref_h * self.shrink or aspect - ref_aspect >= .6

    def _adopt(self, cx, cy, size, now) -> Optional[_State]:
        """Трекер мог выдать новый ID упавшему человеку: наследуем историю у ближайшего потерянного."""
        best, best_d = None, None
        for old_id, old in self.states.items():
            if old.last_seen >= now or not old.hist or now - old.last_seen > 1.5:
                continue
            _, ocx, ocy, ow, oh, _, _ = old.hist[-1]
            dist = math.hypot(cx - ocx, cy - ocy)
            if dist <= max(size, ow, oh) * .9 and (best_d is None or dist < best_d):
                best, best_d = old_id, dist
        return self.states.pop(best) if best is not None else None

    # -- основной вход ------------------------------------------------------ #
    def inspect(self, result, frame_h, now):
        """Возвращает (alerts, detections). alerts: список (тип, описание, уверенность)."""
        alerts, dets = [], []
        boxes = result.boxes
        if boxes is not None and len(boxes):
            names = result.names or {}
            xyxy = boxes.xyxy.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            ids = (boxes.id.cpu().numpy().astype(int).tolist()
                   if boxes.id is not None else [None] * len(xyxy))
            keypoints = getattr(result, "keypoints", None)
            kp = keypoints.data.cpu().numpy() if keypoints is not None else None
            for i in range(len(xyxy)):
                label = str(names.get(int(classes[i]), "unknown"))
                tid = ids[i]
                conf = float(confs[i])
                box = tuple(float(v) for v in xyxy[i])
                if is_fall_label(label):
                    det = Detection(tid, label, conf, box, "fall", "normal")
                    if conf >= self.model_conf:
                        det.state = "fall"
                        self._model_label(det, now, alerts)
                    dets.append(det)
                elif is_person_label(label, kp is not None):
                    det = Detection(tid, label, conf, box, "person", "normal")
                    tilt = self._tilt(kp[i]) if kp is not None and i < len(kp) else None
                    self._person(det, tilt, now, alerts)
                    dets.append(det)
                else:
                    dets.append(Detection(tid, label, conf, box, "other", "normal"))
        self._cleanup(now)
        return alerts, dets

    def _model_label(self, det, now, alerts):
        if det.track is None:
            return
        st = self.states.setdefault(det.track, _State(now))
        st.last_seen = now
        if st.model_since is None or now - st.model_last > .8:
            st.model_since = now
        st.model_last = now
        if now - st.model_since >= self.model_hold and now - st.last_model_alert >= 4.0:
            st.last_model_alert = now
            alerts.append((EV_FALL, f"Модель падений: {det.label} · {det.conf:.0%} · проверьте снимок", det.conf))

    def _person(self, det, tilt, now, alerts):
        x1, y1, x2, y2 = det.box
        w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        aspect, cx, cy = w / h, (x1 + x2) / 2, (y1 + y2) / 2
        lying = self._is_lying(aspect, tilt)
        if det.track is None:                       # без ID временной логики нет
            det.state = "lying" if lying else "normal"
            return
        st = self.states.get(det.track)
        if st is None:
            st = self._adopt(cx, cy, max(w, h), now) or _State(now)
            self.states[det.track] = st
        st.last_seen = now
        was_lying = st.lying_since is not None
        if lying:
            st.unlying_since = None
            if not was_lying:
                st.lying_since = now
                st.transition = self._had_transition(st, now, h, aspect)
                st.alerted = False
        elif was_lying:
            if st.unlying_since is None:
                st.unlying_since = now
            elif now - st.unlying_since >= self.RECOVER_S:
                st.lying_since = st.unlying_since = None
                st.alerted = False
        st.hist.append((now, cx, cy, w, h, aspect, lying))
        while st.hist and now - st.hist[0][0] > self.HISTORY_S:
            st.hist.popleft()

        if st.lying_since is None:
            return
        det.state = "fall" if st.transition else "lying"
        # Человек уже на ногах (льготный период против мерцания детекции) - не тревожим.
        if st.alerted or st.unlying_since is not None:
            return
        duration = now - st.lying_since
        confidence = .55 + .35 * min(duration / 4.0, 1.0)
        if st.transition and duration >= self.hold:
            st.alerted = True
            alerts.append((EV_FALL, f"Резкий переход в положение лёжа · лежит {duration:.1f} с · проверьте снимок",
                           confidence))
        elif not st.transition and duration >= self.long_hold:
            st.alerted = True
            alerts.append((EV_LYING, f"Человек неподвижно лежит {duration:.0f} с · проверьте снимок", .5))

    def _cleanup(self, now):
        for key in [k for k, v in self.states.items() if now - v.last_seen > self.LOST_S]:
            del self.states[key]
