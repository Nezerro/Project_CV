# Campus Watch

Campus Watch is a local video review tool for campus safety experiments. It analyzes a webcam, video file, or RTSP/HTTP stream and records model generated event prompts with snapshots. Prompts are unverified and must be reviewed by a person.

The desktop interface is currently in Russian. It has two tabs: **Мониторинг** (Monitoring) for video analysis and event review, and **ИИ-лаборатория** (AI Lab) for campus-map search, a minimax pursuit game, and event clustering.

## Requirements

- Python 3.10 or newer
- The packages listed in `requirements.txt`
- PyTorch and TorchVision builds compatible with your operating system and hardware; install them using the [official PyTorch instructions](https://pytorch.org/get-started/locally/)
- A camera, video file, or RTSP/HTTP video stream

The app runs its models on CPU by default. A compatible CUDA-enabled PyTorch installation can enable GPU inference when available. The fight model weights download from Hugging Face on first use unless local weights are available. An internet connection is also needed to install dependencies and download model weights.

## Run

From this directory, create and activate a virtual environment, install PyTorch for your machine, then install the project requirements and start the app:

```sh
cd /home/nezerro/Work/campus-watch
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install a compatible PyTorch/TorchVision build: https://pytorch.org/get-started/locally/
python -m pip install -r requirements.txt
python main.py
```

On Windows, activate the environment with `.venv\\Scripts\\activate` instead of the `source` command. If you already have the supplied `.venv`, activate it and install any missing packages.

## Monitor a source

In the **Мониторинг** tab, choose a source type:

- **Камера**: enter the camera index, such as `0`.
- **Видеофайл**: select a video file with **Обзор…**.
- **Поток RTSP/HTTP**: enter a stream URL.

The fall model field accepts a YOLO `.pt` checkpoint. A `best.pt` next to `main.py` is selected automatically when present; otherwise, leaving the field blank uses `yolo11n-pose.pt`. Select **НАЧАТЬ МОНИТОРИНГ** to start and **ОСТАНОВИТЬ** to stop. The sensitivity slider applies while monitoring: higher values can produce earlier or more frequent prompts, including false positives.

## Models and limitations

- **Falls and people on the floor:** the selected YOLO model supplies detections. A pose checkpoint can also use repeated low, horizontal pose estimates as a secondary cue. The bundled `best.pt` recognizes `Fallen`, `Sitting`, and `Standing`; sitting is not a fall.
- **Fight action:** an X3D-M video model evaluates a rolling clip of sampled frames. The app can use a local `final_x3d_realtime.pt` beside `main.py` or the path in `CAMPUS_FIGHT_WEIGHTS`; otherwise it downloads the checkpoint from Hugging Face. The model was trained on RWF-2000 and can confuse sports, play, and aggressive movement. Validate it on representative footage before operational use.
- **Other events:** the supplied weights are not general event sensors. They do not reliably identify fire, weapons, intrusion, medical emergencies, or crowd risks unless you train and validate appropriate models.

The model publisher reports F1 0.93 on its RWF-2000 validation split. That result describes the publisher's benchmark, not accuracy on campus cameras. See the [model card](https://huggingface.co/visionlab-ai/school-violence-detection-models) and [publisher preprocessing code](https://huggingface.co/spaces/visionlab-ai/school-violence-detection-app/blob/main/video_utils.py).

## Events and privacy

Events appear newest first with a timestamp, score, details, and snapshot. The journal is stored as `events.json` and snapshots as JPEG files in the operating system's application data directory. The journal retains up to 1,000 events, and the table shows up to 200. Use **Снимки** to open the snapshot folder, double-click an event to open its image, and **CSV** to export the journal (including snapshot paths).

Video is processed on the device. The app does not record continuous video, use audio, or perform face recognition. Event snapshots and the journal remain on the computer unless you copy or export them. Use only cameras and footage you are authorized to process, and set appropriate access and retention controls for saved images.

## AI Lab

The **ИИ-лаборатория** tab demonstrates algorithms using a fixed campus map: BFS, DFS, and A* route search; minimax with optional alpha-beta pruning for a guard-and-intruder game; and k-means clustering over logged event time and type. The clustering panel can use sample data or the current event journal. Map, game, and sample clustering results are demonstrations, not live dispatch or risk predictions.

## Fine-tuning

For better performance on a specific site, collect footage you are authorized to use from the actual camera positions. Label positive examples and hard negatives such as sports, play, sitting, crowds, and people carrying objects. Keep test footage separated by camera and recording session. Fine-tune a detection checkpoint with Ultralytics and evaluate class-wise precision, recall, and alert counts on held-out footage:

```sh
yolo detect train model=best.pt data=/path/to/campus.yaml epochs=100 imgsz=640 batch=16 patience=20
```

The supplied fall checkpoint uses the labels `Fallen`, `Sitting`, and `Standing`. Keep labels consistent with the app's expected classes when extending or replacing it. Fight recognition depends on motion over time; training a frame detector alone does not replace a temporal action model. Review checkpoint and dataset licenses before institutional use.
