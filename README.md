<h1 align="center">⚡Real-time Human Re-Identification⚡<br><i></i></h1>

<p align="center">Build with</p>
<p align="center">
  <img src="https://img.shields.io/badge/-Apache Kafka-05122A?style=for-the-badge&logo=apachekafka"/>
  <img src="https://img.shields.io/badge/-Apache Spark-05122A?style=for-the-badge&logo=apachespark"/>
  <img src="https://img.shields.io/badge/-Flask-05122A?style=for-the-badge&logo=flask"/>
  <img src="https://img.shields.io/badge/-Python-05122A?style=for-the-badge&logo=python"/>
</p>

## Introduction

This is a practical Project of using multiple technologies as Apache Kafka, Apache Spark, with simple Deep Learning Models like YOLOv5 (for Human Detection) and ... (For Person Re-identification) to address Human Re-identification in real-time.

The scenario is that there are multiple Cameras in a building, we want to detect people appear in these cameras and identify them with unique ID. A person can appear in one camera at this point and re-appear in another camera later, our goal is to identify this person in these cameras with the same ID (a.k.a Person Re-identification task).

## Prerequisites

- `Python 3.11`
- `Apache Spark (>= 3.5.0)`
- `Apache Kafka (>= 3.6.0)`

Note: 
- I tested this on `python=3.12` (on November 2023) but it didn't work. You can test it on latest version, if it still doesn't work, I recommended using `python=3.11`.

## Getting Started
- Install Apache Spark and Apache Kafka (Remember to test if it works).

- Clone this repository

- Create a Python environment and install necessary packages
  
```bash
conda create -n human_reid python=3.11
conda activate human_reid
pip install -r requirements.txt
```

## 📖 Usage

- Run `Zookeeper`
```
/path/to/your/kafka/bin/zookeeper-server-start.sh /path/to/your/kafka/config/zookeeper.properties
```
- Run `Kafka`
```
/path/to/your/kafka/bin/kafka-server-start.sh /path/to/your/kafka/config/server.properties
```
- Run the  Consumer (our Flask Server) to start listening and receiving data from our Kafka Producer.
```bash
python Consumer.py
```

- Go to (default to) `localhost:5000` to check if our Flask Server is running successfully.

- Run Producers to publish recorded footages from our cameras to their *topics* (run each producer separately).
```bash
python Producer.py --topic topic_camera_00 --camera ./videos/camera_00.mp4
```
```bash
python Producer.py --topic topic_camera_01 --camera ./videos/camera_01.mp4
```

- Now Refresh `localhost:5000`, you should see both footages are displayed on the page. 

## About Us

We are a Group of Student majored in Data Science at **University of Information Technology (UIT), VNU-HCM**.
