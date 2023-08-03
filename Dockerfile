FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN mkdir -p /data/generated/

COPY ./requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

COPY ./data/generated/ /data/generated
