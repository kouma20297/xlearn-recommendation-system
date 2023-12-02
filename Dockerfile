# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install system dependencies
RUN apt-get update && \
    apt-get install -y cmake

# Install Python dependencies
RUN pip install --no-cache-dir pandas -r requirements.txt
RUN pip install --upgrade xlearn

# Create a directory for training data if it doesn't exist
RUN mkdir -p 学習用データ

# Copy training data into the container
COPY 学習用データ/ml-10M100K /app/学習用データ/

# Run app.py when the container launches
CMD ["python", "main.py"]
