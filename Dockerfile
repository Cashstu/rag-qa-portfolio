FROM python:3.12-slim

WORKDIR /srv/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY data/raw/ data/raw/
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENV PORT=8080
EXPOSE 8080

ENTRYPOINT ["./entrypoint.sh"]
