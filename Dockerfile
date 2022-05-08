# Dockerfile

FROM python:3.8.8-slim-buster

RUN pip install mlflow>=1.0 \
    && pip install numpy \
    && pip install pandas \
    && pip install scikit-learn \
    && pip install lightgbm \
    command: "python train.py"