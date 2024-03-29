FROM python:3.10-slim

RUN apt-get update && \
  apt-get upgrade -y && \
  apt-get install -y --no-install-recommends git build-essential && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install numpy Cython
RUN pip install -r /app/requirements.txt

WORKDIR /app
COPY . /app

ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
CMD ["python", "scripts/run_step.py"]
