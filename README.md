<h1 align="center">⚡Real-Time Person Re-ID in Multi-Camera Surveillance Systems⚡<br></h1>

<p align="center">Build with</p>
<p align="center">
  <img src="https://img.shields.io/badge/-Apache Kafka-05122A?style=for-the-badge&logo=apachekafka"/>
  <img src="https://img.shields.io/badge/-Apache Spark-05122A?style=for-the-badge&logo=apachespark"/>
  <img src="https://img.shields.io/badge/-Flask-05122A?style=for-the-badge&logo=flask"/>
  <img src="https://img.shields.io/badge/-Python-05122A?style=for-the-badge&logo=python"/>
</p>

## Introduction

This project leverages a combination of cutting-edge technologies, including Apache Kafka and Apache Spark, along with straightforward Deep Learning Models like YOLOv5 (for *Human Detection*) and ResNet (for *Person Re-identification*). The primary objective is to tackle the challenge of *Person Re-identification* in real-time within a given scenario.

In the context of a building equipped with multiple cameras, the aim is to detect individuals across these camera feeds and assign unique IDs to each person. Notably, a person may initially appear in one camera and subsequently reappear in another. The overarching objective is to seamlessly identify the same individual across these cameras by associating them with a consistent ID—a task commonly referred to as *Person Re-identification*.

## Prerequisites

- `Python 3.11`
- `Apache Spark (>= 3.5.0)`
- `Apache Kafka (>= 3.6.0)`

Note: 
- I tested this on `python=3.12` (November 2023) but it didn't work. You can test it on the latest version, if it still doesn't work, I recommend using `python=3.11`.

## Getting Started
- Install Apache Spark and Apache Kafka (Remember to test if it works).

- Clone this repository

- Create a Python environment and install the necessary packages
  
```bash
conda create -n human_reid python=3.11
conda activate human_reid
pip install -r requirements.txt
```

## Usage

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

- Go to (default to) [`localhost:5000`](localhost:5000) to check if our Flask Server is running successfully.

- Run Producers to publish recorded footage from our cameras on these *topics* (run each producer separately).
```bash
python Producer.py --topic topic_camera_00 --camera ./videos/camera_00.mp4
```
```bash
python Producer.py --topic topic_camera_01 --camera ./videos/camera_01.mp4
```

- Now Refresh [`localhost:5000`](localhost:5000), you should see both footages displayed on the page. 

## Contribution
This repo is continuously fixed and updated over time, feel free to fix some issues/bugs, all are welcome here!

## About Us

We are a Group of Students majoring in Data Science at **University of Information Technology (UIT), VNU-HCM**.
