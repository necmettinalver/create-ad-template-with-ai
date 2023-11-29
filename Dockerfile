# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/engine/reference/builder/

ARG PYTHON_VERSION=3.9.12
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

RUN pip install --upgrade pip==23.3.1

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

RUN useradd -m huggingface

USER huggingface

WORKDIR /home/huggingface

# Switch to the non-privileged user to run the application.
RUN mkdir -p /home/huggingface/.cache/huggingface \
  && mkdir -p /home/huggingface/input \
  && mkdir -p /home/huggingface/output

# Copy the source code into the container.
COPY . .

USER root

RUN chown -R huggingface:huggingface /home/huggingface/app

USER huggingface

# Expose the port that the application listens on.
EXPOSE 8000

# Run the application.
CMD uvicorn 'app.main:app' --host=0.0.0.0 --port=8000
