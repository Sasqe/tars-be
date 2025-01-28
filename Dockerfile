# Use the official Python image
FROM python:3.12.8

# Set the working directory in the container
WORKDIR /app

# Copy application files into the container
COPY . .

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 vim && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install fastapi[all] --upgrade





# Expose the application port
EXPOSE 8000

# Set environment variables (optional)
ENV NAME Tars

# Command to run your application
CMD ["uvicorn", "tars:app", "--host", "0.0.0.0", "--port", "8000"]
