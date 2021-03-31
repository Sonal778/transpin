FROM tensorflow/tensorflow

RUN apt-get update

COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY . .

EXPOSE 5000

EXPOSE $PORT

WORKDIR ./Server

CMD ["python", "server.py"]
