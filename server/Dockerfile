
#FROM nvcr.io/nvidia/pytorch:19.12-py3
FROM adujardin/detectron2:torch1.8-cu111
RUN apt-get update -y
RUN apt-get install gcc -y
RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y
#RUN apt-get install -y python3-pip python3-dev build-essential
COPY . /app
WORKDIR /app

RUN pip install fastapi
RUN pip install uvicorn
RUN pip install nest_asyncio 
RUN pip install opencv-python

ENV PYTHONIOENCODING=UTF-8
EXPOSE 5000
ENTRYPOINT ["python3"]
CMD ["tiger_serv.py"]