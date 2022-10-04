# FROM tensorflow/tensorflow:2.1.0-gpu-py3
FROM tensorflow/tensorflow:2.7.1-gpu

COPY ./ /src
RUN pip install -r /src/requirements.txt
