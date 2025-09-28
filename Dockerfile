# Use the official Python image
FROM python:3.12.8-slim

# Set the working directory
WORKDIR /app

# Copy dependency files first (better caching)
COPY requirements.txt .

# Install system dependencies for OpenCV (and friends)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    vim \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir fastapi[all] --upgrade

# Copy application files last
COPY . .

# Expose the app port for Cloud Run
EXPOSE 8080

# Run the app (PORT comes from env, defaults to 8000 in code)
CMD ["python", "tars.py"]