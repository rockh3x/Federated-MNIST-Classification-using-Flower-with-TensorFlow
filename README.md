🧠 Phase 2 – Your Program Description
We worked on:
Title:
🚀 Federated MNIST Classification using Flower with TensorFlow
Goal:
Set up a basic federated learning simulation using the Flower framework, with:

A CNN model built in TensorFlow/Keras
Non-distributed (single-machine) simulation
Each client training on a local MNIST subset
Accuracy plotted per round

Structure:

model.py – CNN definition
client.py – Flower client with fit, evaluate, and get_parameters
server.py – Custom server strategy (LoggingFedAvg) with accuracy logging
run_phase2.py – Script to simulate clients
Used default IID data splits (in Phase 2)


This was the base we tested and successfully ran. It gave us a strong foundation to build:

Phase 3 – Accuracy plotting, strategy tuning
Phase 4 – Realistic non-IID simulation with partitioning and expanded metrics
