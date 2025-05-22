FROM python:3.12-slim

ENV PYTHONUNBUFFERED True

WORKDIR /app

# Copy everything except what's in .dockerignore
COPY . .

# Create virtual environment inside the container
RUN python -m venv venv

# Activate the venv and install dependencies
RUN ./venv/bin/pip install --upgrade pip && \
    ./venv/bin/pip install --no-cache-dir -r requirements.txt

# Use the venv by default
ENV PATH="/app/venv/bin:$PATH"

CMD ["python", "askypembed.py"]