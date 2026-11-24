# Campus Watch

Campus Watch is a local video review tool for campus safety experiments. It analyzes a webcam, video file, or RTSP stream and stores event prompts with snapshots. A prompt is an unverified model output and must be reviewed by a person.

## Start

```sh
cd /home/nezerro/Work/campus-watch
source .venv/bin/activate
python main.py
```

If you need a new environment, install the packages in `requirements.txt` plus a PyTorch build compatible with your GPU. On first launch, the fight model downloads from Hugging Face and is cached. The fall field automatically selects `best.pt` beside `main.py`; otherwise pick a YOLO checkpoint or leave it blank to use `yolo11n-pose.pt`.

Select Camera and source `0`, `1`, etc.; Video file and enter a path; or RTSP and enter a stream URL. Use the sensitivity slider to trade earlier prompts for more false positives. The default Very High mode is intended for tests. It lowers detection thresholds and cooldown, but cannot improve the underlying checkpoint's learned accuracy.

## Detectors and limitations

- **Falls:** the selected YOLO checkpoint provides fall labels when present. If it is a pose model, repeated low horizontal pose estimates provide a secondary cue. The popular `best.pt` supplied for this project is a three-class fall model (Fallen/Sitting/Standing), so sitting must not be treated as a fall.
- **Fight action:** a separate X3D-M video model evaluates 16 uniformly sampled frames from a rolling clip. Preprocessing follows the model publisher's companion inference code (RGB, 224×224, mean 0.45, standard deviation 0.225). It was trained on RWF-2000 and may confuse sports, play, and aggressive movement; test clips from your own cameras before use.
- **Other campus events:** the supplied weights are not general event sensors. They do not reliably identify fire, weapons, intrusion, medical emergencies, or crowd risks unless you train and validate a checkpoint for those classes.

Very High sensitivity intentionally favors early detection and can create noisy alerts. The fight publisher lists F1 0.93 on its RWF-2000 validation split; that is the publisher's benchmark on that dataset, not a campus accuracy guarantee. ([Model card](https://huggingface.co/visionlab-ai/school-violence-detection-models), [publisher preprocessing code](https://huggingface.co/spaces/visionlab-ai/school-violence-detection-app/blob/main/video_utils.py)).

## Event review and privacy

Events are shown newest-first with time, score, details, and a snapshot button. A JSON event log and JPEG snapshots are stored under the operating system's Campus Watch application data folder. CSV export includes snapshot paths. The app does not record continuous video, use audio, or perform face recognition. Use only cameras and footage you are authorized to process, and set appropriate access and retention controls for saved images.

## Fine-tune

For better campus performance, collect authorized footage from the actual camera positions and label both positive events and hard negatives (sports, play, sitting, crowds, carrying objects). Keep test footage separated by camera and recording session. Fine-tune a detection checkpoint using Ultralytics, then evaluate class-wise precision/recall and alert counts on held-out footage:

```sh
yolo detect train model=best.pt data=/path/to/campus.yaml epochs=100 imgsz=640 batch=16 patience=20
```

The dataset class names must match the labels expected by the app. The fall checkpoint uses `Fallen`, `Sitting`, `Standing`; extending it requires consistent labels for every added class. Fight recognition depends on motion over time, so training a frame detector alone is not a substitute for a temporal action model. Review the checkpoint and dataset license before institutional use.
