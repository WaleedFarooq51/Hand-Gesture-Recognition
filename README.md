# Human Hand Gesture Recognition (Sign Language Detection)

This repository implements a **real-time hand gesture and sign-language alphabet recognition system** using **TensorFlow Object Detection API** and **Transfer Learning** (MobileNet SSD V2). The project captures hand gestures via a webcam (OpenCV), detects and localizes hands, and maps detected gestures to alphabet letters or common sign gestures. This README summarizes the project report by Shammas Javed and Muhammad Waleed Khan. fileciteturn1file0

---

## 🔎 Project Overview

The goal of this project is to enable communication between deaf/dumb users and the general public by translating hand gestures (including the American Sign Language alphabet and selected common gestures) into readable text in real time. The system uses a combination of data annotation, TFRecord creation, and transfer learning with the **MobileNet SSD (V2)** object detector to achieve fast, mobile-friendly inference.

Key highlights:
- Real-time detection via webcam using OpenCV.
- Transfer learning from a COCO-pretrained MobileNet SSD (V2) backbone (320×320 input resolution).
- Training and inference using TensorFlow Object Detection API.
- Dataset is a mix of pre-annotated images and custom-captured webcam images; annotations created with **LabelMe**. fileciteturn1file0

---

## 📁 Suggested Repository Structure

```
.
├─ data/
│  ├─ images/
│  │  ├─ train/                # training images
│  │  └─ test/                 # testing images
│  ├─ annotations/             # LabelMe JSON / XML annotations
│  ├─ train.record             # TFRecord for training
│  └─ test.record              # TFRecord for evaluation
├─ models/                     # model checkpoints and exported models
├─ scripts/
│  ├─ generate_tfrecord.py     # convert annotations -> TFRecord
│  ├─ train.py                 # wrapper to launch TF OD training
│  ├─ export_model.py          # export saved_model / frozen graph
│  └─ infer_webcam.py          # real-time inference script using OpenCV
├─ pipeline.config             # TF OD pipeline config (tuned for MobileNet SSD V2)
├─ label_map.pbtxt             # label map (A-Z + custom gestures)
├─ notebooks/
│  └─ hand_gesture_detection_colab.ipynb   # recommended Colab notebook
├─ README.md
└─ requirements.txt
```

---

## 🧩 Dataset & Annotation

- The dataset contains images for **26 alphabet signs (A–Z)** and additional custom gestures (e.g., *I love you*, *goodness*, *bad*, etc.). Some data were pre-annotated; other images were captured with a webcam and annotated manually using **LabelMe**. fileciteturn1file0
- Annotations include bounding boxes (coordinates) and class labels.
- A `label_map.pbtxt` file maps numeric class IDs to human-readable labels (first 26 classes for alphabets).

**Annotation tips:**
- Use consistent labeling and bounding-box conventions (tight boxes around hands).
- For best performance, collect varied backgrounds, lighting conditions, hand orientations, skin tones, and clothing.
- Export annotations into the format expected by `generate_tfrecord.py` (CSV or LabelMe JSON → TFRecord).

---

## 🛠 Tools & Frameworks

- Python 3.8+
- TensorFlow 2.x (Object Detection API v2)
- OpenCV (cv2) for webcam capture and display
- LabelMe for annotation (or any bounding-box annotation tool)
- `protoc` for compiling TF record utilities (if required)
- Optional: Google Colab GPU runtime (recommended for training)

Add these to `requirements.txt` (minimum):
```
tensorflow>=2.6
opencv-python
labelme
numpy
pandas
Pillow
matplotlib
```
*(Exact TF version should match the OD API checkpoints used; freeze with `pip freeze > requirements.txt`.)*

---

## 🔧 Model & Training Summary

**Model architecture:** MobileNetV2 + SSD (pretrained on COCO 2017, input 320×320). Transfer learning fine-tunes the detection head on the custom hand-gesture dataset.

**Training configuration (as used in experiments):**  
- Batch size: **4** (kept small for faster convergence on limited GPU/Colab resources). fileciteturn1file0
- Optimizer & learning rate: Tuned in `pipeline.config` (use conservative LR for transfer learning).  
- Training platform: Google Colab GPU (recommended). fileciteturn1file0

**TFRecord & label map:** Create `train.record` and `test.record` and update `label_map.pbtxt`. The OD API `pipeline.config` must point to these TFRecords and the checkpoint initializer (MobileNet SSD checkpoint).

---

## ✅ Results & Evaluation

- Two types of inference were run: (1) image-based testing on held-out images, and (2) **real-time webcam inference**. The model returns bounding boxes with class labels and confidence scores. fileciteturn1file0
- **Performance:** Many alphabet classes achieved confidence rates **> 80%** in the reported experiments; some classes & gestures showed confidence **< 70%** and confusion with similar signs. See the report for per-class confidence plots. fileciteturn1file0

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

## 📌 Practical Tips & Recommendations

- **Augmentation:** Apply random flips, rotations, brightness/contrast, occlusion augmentation, and random crops to improve robustness.
- **Background subtraction / preprocessing:** Removing background clutter, adaptive histogram equalization, or simple cropping around hands can boost accuracy.
- **More data:** Collecting multi-angle samples and diverse users (different skin tones, clothing, backgrounds) will likely yield the largest performance gains.
- **Use class weighting or focal loss** if you have skewed data across classes (some letters/gestures rarer than others).

---

## 🧾 Files & Scripts to Add (Recommended)

- `scripts/generate_tfrecord.py` — convert annotations → TFRecord
- `scripts/train.py` — wrapper to start training and manage checkpoints
- `scripts/export_model.py` — to export saved_model/frozen_graph
- `scripts/infer_image.py` — batch inference on images
- `scripts/infer_webcam.py` — webcam real-time demo (OpenCV)
- `notebooks/hand_gesture_detection_colab.ipynb` — full Colab notebook (data prep → train → infer)

**Recommended Colab filename:** `hand_gesture_detection_colab.ipynb` or `01_hand_gesture_detection_colab.ipynb` for a multi-notebook repo.

---

## 🧑‍💻 Authors & Contact

- **Shammas Javed** — NUST, SEECS — <sjaved.msee21seecs@seecs.edu.pk>.
- **Muhammad Waleed Khan** — NUST, SEECS — <mwkhan.msee21seecs@seecs.edu.pk>. fileciteturn1file0

---

## 🧭 Roadmap & Future Work

- Increase dataset size and diversity (multi-user, multi-background).
- Explore stronger detection backbones or two-stage detectors (e.g., Faster R-CNN) for improved per-class accuracy (tradeoff: speed).
- Add sequence-level modeling (temporal smoothing / RNN) to disambiguate tricky gestures that are temporally distinct.
- Build a mobile-friendly deployment (TensorFlow Lite) for on-device inference.

---

## 📄 License & Citation

Include a LICENSE file (MIT/Apache-2.0 recommended).

If you use this work, please cite the accompanying project report:
```bibtex
@techreport{hand_gesture_recognition_2025,
  title = {Human Hand Gesture Recognition},
  author = {Shammas Javed and Muhammad Waleed Khan},
  year = {2025},
  institution = {NUST - SEECS}
}
```

---

## Acknowledgements & References

This README is based on the project report by Shammas Javed and Muhammad Waleed Khan (NUST, SEECS). For details on experiments, per-class confidence scores, and architecture diagrams, see the original report. fileciteturn1file0
