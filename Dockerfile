FROM python:3.11-slim

WORKDIR /workspace

ENV MUJOCO_PATH=/workspace/mujoco

COPY requirements.txt .
RUN apt-get update && apt-get install -y swig build-essential
RUN pip install --no-cache-dir -r requirements.txt
COPY . /workspace
