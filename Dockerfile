FROM tensorflow/tensorflow:2.1.0-gpu-py3

COPY ./ /src
RUN pip install -r /src/requirements.txt