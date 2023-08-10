FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN mkdir -p /data/generated/
RUN mkdir -p /data/annotated/

COPY ./requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

COPY ./data/annotated/ /data/annotated
COPY ./data/generated/ /data/generated
