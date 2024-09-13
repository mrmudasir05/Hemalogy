# Use the official Python 3.12.3 image as the base image
FROM python:3.12.3-slim

# Set the working directory in the container
WORKDIR /app

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
CMD ["python", "app.py"]
