# 1D-CNN-UAV-Predictive-Maintenance
1D-CNN Deep learning model trained on real UAV flight sensor readings, sensors Including Acce, Baro, Mag, GPS, Gyro. An open source dataset from RflyMAD obtained from PX4 embedded quadcopters.
Overview
This project implements a complete Fault Detection and Diagnostics (FDD) pipeline for PX4-based multirotor UAVs. It processes raw PX4 ULog flight logs through a preprocessing pipeline and runs inference using a trained 1D Convolutional Neural Network to classify sensor health into six categories.
No specialist knowledge is required to use the tool a pilot provides a flight log file and receives a visual health report showing exactly which sensor, if any, behaved anomalously during the flight.
Dataset.

Trained on the RflyMAD Real-Flight Dataset by Beihang University.

286 real PX4 flight cases
5 sensor fault types x 5 flight modes
Real hardware flights with controlled fault injection
Ground truth labels from rfly_ctrl_lxl uORB topic

Dataset: https://rfly-openha.github.io/documents/4_resources/dataset.html

