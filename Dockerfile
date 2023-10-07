FROM python:3.10.9-bullseye
WORKDIR .
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
