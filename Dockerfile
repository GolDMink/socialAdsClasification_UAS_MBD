FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 7860

CMD streamlit run streamlit_app.py \
    --server.address=0.0.0.0 \
    --server.port=7860 \
    --server.baseUrlPath=/ 