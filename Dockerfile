FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8888 8501

CMD export HOME=/app && \
    export JUPYTER_CONFIG_DIR=/app/.jupyter/config && \
    export JUPYTER_DATA_DIR=/app/.jupyter/data && \
    export JUPYTER_RUNTIME_DIR=/app/.jupyter/runtime && \
    nohup jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --IdentityProvider.token='' > logs/jupyter.log 2>&1 & \
    streamlit run app.py --server.port=8501 --server.address=0.0.0.0
