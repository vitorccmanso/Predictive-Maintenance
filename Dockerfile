FROM python:3.11
WORKDIR /pred_app
COPY /requirements.txt /pred_app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /pred_app/requirements.txt
COPY /app /pred_app/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]