FROM python:3.6-stretch

RUN apt update -y
RUN apt install -y xvfb
RUN pip install gym
RUN apt install -y python-opengl
RUN pip install pillow
RUN pip install chainer
RUN pip install dask distributed

WORKDIR /workdir

ADD ./test.sh ./test.sh
ADD ./neuroevolution ./neuroevolution
ADD ./main.py ./main.py
