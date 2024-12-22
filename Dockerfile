FROM python:3.12-slim-bookworm

RUN mkdir /opt/anomaly-detector-api

WORKDIR /opt/anomaly-detector-api

COPY app /opt/anomaly-detector-api/app/
COPY model /opt/anomaly-detector-api/model/

RUN pip install -r app/requirements.txt

ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=8000
ENV MODEL_FILE='/opt/anomaly-detector-api/model/anomaly_detector_pipeline.pkl'

EXPOSE 8000

CMD python app/anomaly-detector.py