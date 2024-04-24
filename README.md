# Concrete-Damage-Detector

<p align="justify">Traditional visual inspections for concrete damage are time-consuming, expensive, and prone to human error. Inspectors may miss hidden cracks or misjudge their severity, leading to delayed repairs and potential structural failures. This project proposes an innovative solution using deep learning to automate concrete damage detection. It leverages the YOLOv8 deep learning model. Known for its balance between accuracy and speed, YOLOv8 is ideal for real-time detection, enabling rapid assessments during inspections. A custom dataset is central to this project. This dataset comprises a diverse collection of images containing various damage types and severities on concrete surfaces.Traditional visual inspections for concrete damage are time-consuming, expensive, and prone to human error. Inspectors may miss hidden cracks or misjudge their severity, leading to delayed repairs and potential structural failures.</p>

## Train YOLO-v8 model on a custom dataset using [Roboflow](https://roboflow.com/)

- Importing Libraries
```
from ultralytics import YOLO
from roboflow import Roboflow
```
- Initializing Roboflow client
```
rf = Roboflow(api_key="YOUR_KEY")
```
- Specifying workspace, dataset and version
```
project = rf.workspace("YOUR_WORKSPACE").project("YOUR_DATASET")
version = project.version(1)
```
- Download dataset (You will find a data.yaml file in the folder created which you can use to train your model in next step)
```
dataset = version.download("yolov8")
```
- Install model and train it on your dataset
```
model  = YOLO('yolov8x-seg.pt')
model.train(data='data.yaml', epochs=10)
model.val()
```
- After training is done, you can find a "best.pt" weight file which you can use as your deep learning model

<br>
<details>

<summary>Credits</summary>

- [Ultralytics](https://docs.ultralytics.com/)

- [Roboflow](https://roboflow.com/)

- [Supervision](https://supervision.roboflow.com/latest/)
  
</details>
