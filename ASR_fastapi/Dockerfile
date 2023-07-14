FROM python:3.9


WORKDIR /ASR_fastapi


COPY ./requirements.txt /ASR_fastapi/requirements.txt


RUN pip install --no-cache-dir --upgrade -r /ASR_fastapi/requirements.txt

COPY ./app /ASR_fastapi/app


CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
