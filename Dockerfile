FROM python:3.6
COPY requirements.txt /app/requirements.txt
RUN pip install numpy Cython
RUN pip install -r /app/requirements.txt

WORKDIR /app
COPY . /app

WORKDIR /app/scripts

ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
CMD ["python", "run_step.py"]
