FROM python:3.9


WORKDIR /classification_fastapi


COPY ./requirements.txt /classification_fastapi/requirements.txt


RUN pip install --no-cache-dir --upgrade -r /classification_fastapi/requirements.txt

COPY ./app /classification_fastapi/app


CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
