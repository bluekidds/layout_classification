# app/Dockerfile

FROM python:3.9-slim

EXPOSE 8051

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

RUN pip install layoutparser torchvision && pip install "git+https://github.com/facebookresearch/detectron2.git@v0.5#egg=detectron2"


RUN pip install pip install "layoutparser[ocr]"

#RUN git clone https://github.com/bluekidds/layout_classification.git .

RUN pip install "pinferencia"

RUN git clone https://github.com/bluekidds/layout_classification.git .

RUN pip3 install -r requirements.txt

#RUN pinfer app:service --reload

#ENTRYPOINT ["uvicorn", "app.app:service", "--reload"]
ENTRYPOINT ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
