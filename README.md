# Human Hand Gesture Recognition (Sign Language Detection)

This repository implements a **real-time sign-language recognition system** using **TensorFlow Object Detection API**. The project captures hand gestures (including the American Sign Language alphabet) via a webcam (OpenCV), detects and localizes hands, and maps detected gestures to alphabet letters or common gestures. 

---

## 🔎 Key Highlights

- Real-time detection via webcam using OpenCV.
- Transfer learning from a COCO-pretrained MobileNet SSD (V2) backbone.
- Training and inference using TensorFlow Object Detection API.
  
---

## 📁 Suggested Repository Structure

```
.
├─ workspace/
│  ├─ images/
│  │  ├─ train/                # training images
│  │  └─ test/                 # testing images
│  ├─ annotations/            
│  │  ├─ label_map.pbtxt       # LabelMe JSON / XML annotations
│  │  ├─ train.record          # TFRecord for training
│  │  └─ test.record           # TFRecord for evaluation
│  ├─ models/                  # trained model checkpoints
│  │  ├─ my_ssd_mobnet
│  ├─ pre-trained models       # pre-trained SSD-MobNet model checkpoints    
├─ scripts/
│  ├─ generate_tfrecord.py     # convert annotations -> TFRecord
├─ README.md
└─ requirements.txt
```

---

## 🧩 Dataset & Annotation

- The dataset contains images for **26 alphabet signs (A–Z)** and additional custom gestures. Some data were pre-annotated; other images were captured with a webcam and annotated manually using **LabelMe**.
- Annotations include bounding boxes (coordinates) and class labels.
- A `label_map.pbtxt` file maps numeric class IDs to human-readable labels.
  
---

## 🛠 Tools & Frameworks

- Python 3.8+
- TensorFlow 2.x (Object Detection API v2)
- OpenCV (cv2) for webcam capture and display
- LabelMe for annotation (or any bounding-box annotation tool)
- `protoc` for compiling TF record utilities

---

## 🔧 Model architecture

MobileNetV2 + SSD (pretrained on COCO 2017, input 320×320). Transfer learning fine-tunes the detection head on the custom hand-gesture dataset.

---

## ✅ Results & Evaluation

- Two types of inference were run: (1) image-based testing on held-out images, and (2) **real-time webcam inference**.

- The model returns bounding boxes with class labels and confidence scores.

![Sample Detection](./results/img1.png) ![Sample Detection](./results/img2.png) ![Sample Detection](./results/img3.png)

**Limitations observed:**
- Uncontrolled backgrounds and low-light conditions reduce accuracy.
- Varied skin tones, clothing, partial occlusions, and face presence can confuse the detector.
- Some similar letters or gestures are confused with each other — remedyable with more data and augmentation. fileciteturn1file0

---

## 🚀 Quickstart — Train + Inference (Colab / Local)

> **Recommend:** Use Google Colab with GPU runtime for training (`Runtime` → `Change runtime type` → GPU).

### 1) Clone repo
```bash
git clone <your-repo-url>
cd <repo-name>
```

### 2) Prepare dataset & label map
- Put images into `data/images/train` and `data/images/test`.
- Ensure your `label_map.pbtxt` lists all classes (A–Z and custom gestures).

### 3) Generate TFRecords
Example (replace arguments with your script’s CLI):
```bash
python scripts/generate_tfrecord.py --annotations_dir data/annotations --images_dir data/images/train --output train.record --label_map label_map.pbtxt
python scripts/generate_tfrecord.py --annotations_dir data/annotations --images_dir data/images/test --output test.record --label_map label_map.pbtxt
```

### 4) Configure `pipeline.config`
- Set `fine_tune_checkpoint` to the path of the downloaded MobileNet SSD V2 checkpoint.
- Set `train_input_reader` / `eval_input_reader` to your `train.record` / `test.record`.
- Adjust `batch_size`, `num_classes`, learning rate schedule, and `num_steps` as needed.

### 5) Train (TF OD API example)
```bash
# TF2 OD API (example; path to model_main_tf2.py may vary)
python /path/to/models/research/object_detection/model_main_tf2.py     --model_dir=models/my_mobilenet_ssd_v2     --pipeline_config_path=pipeline.config
```

### 6) Export trained model (SavedModel)
```bash
python /path/to/models/research/object_detection/exporter_main_v2.py     --input_type image_tensor     --pipeline_config_path pipeline.config     --trained_checkpoint_dir models/my_mobilenet_ssd_v2     --output_directory models/exported/saved_model
```

### 7) Run real-time inference (webcam)
```bash
python scripts/infer_webcam.py --model_dir=models/exported/saved_model --label_map label_map.pbtxt
```

`infer_webcam.py` should:
- Load the exported `saved_model` with `tf.saved_model.load()`,
- Capture frames from OpenCV (`cv2.VideoCapture(0)`),
- Preprocess frames to model input (resize to 320×320, normalize if required),
- Run inference and apply non-max suppression / draw boxes,
- Display label + confidence and optionally save text output to a log file.

---


