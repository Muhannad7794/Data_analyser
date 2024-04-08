FROM python:3

COPY ./requirments.txt /requirments.txt
RUN pip install --no-cache-dir -r /requirments.txt

WORKDIR /Data_analyser_Dockerized
COPY . /Data_analyser_Dockerized/