FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

RUN mkdir -p /data/generated/
RUN mkdir -p /data/annotated/

RUN apt-get update
RUN apt-get install vim -y
COPY ./requirements.txt /tmp/

COPY ./data/annotated/ /data/annotated
COPY ./data/generated/ /data/generated

RUN pip install -r /tmp/requirements.txt
RUN pip install einops

RUN mkdir -p /tmp/pointnet2_ops_lib/
COPY ./models/pointnet2_ops_lib/ /tmp/pointnet2_ops_lib/
RUN pip install --upgrade pip
RUN pip install /tmp/pointnet2_ops_lib/.
