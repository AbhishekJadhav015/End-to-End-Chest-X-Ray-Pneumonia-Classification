# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for OpenCV/image processing if needed
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies (the --no-cache-dir flag shrinks the final image size)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of application code and the artifacts folder
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Add a healthcheck so cloud providers know the app is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Command to run the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]