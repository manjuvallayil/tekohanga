FROM python:3.8
COPY . /app
WORKDIR /app/web_app/app
RUN pip install -r requirements.txt
EXPOSE $PORT
CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT run_app:app