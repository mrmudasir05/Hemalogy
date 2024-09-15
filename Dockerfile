FROM python:3.12.3-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for OpenCV and other libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    xvfb \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file into the container
COPY requirements.txt requirements.txt

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Set environment variables for Dash
ENV DASH_DEBUG_MODE='false'
ENV PORT=8080

# Expose the required port for the app to run
EXPOSE 8080

# Define the command to run the app
CMD ["gunicorn", "app:server", "-b", "0.0.0.0:8080"]