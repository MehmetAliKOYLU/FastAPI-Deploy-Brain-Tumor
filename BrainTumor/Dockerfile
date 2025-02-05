FROM python:3.12.7

WORKDIR /usr/local//src/app

COPY . .

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --no-cache-dir -r requirements.txt


EXPOSE 80

CMD [ "python","App/main.py" ]