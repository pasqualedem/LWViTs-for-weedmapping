FROM nvidia/cuda:11.4.0-base-ubuntu20.04
COPY requirements.txt requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get update -y && apt-get install -y python3 && apt install -y python3-pip && python3 -m pip install --upgrade pip && python3 -m pip install -r requirements.txt
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

WORKDIR /LWViTs-for-weedmapping
COPY . /LWViTs-for-weedmapping
EXPOSE 8000

RUN python3 wd.py download
RUN python3 wd.py preprocess --subset Sequoia
RUN python3 wd.py preprocess --subset Sequoia
RUN python3 wd.py augment --subset Sequoia
RUN python3 wd.py augment --subset Sequoia

CMD ["python3", "wd.py", "experiment", "--file", "./params/RedEdge/SplitLawin.yaml"]
