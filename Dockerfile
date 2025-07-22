# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy dependency files first
COPY requirements.txt .

# Install only what you need
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code and model files
COPY . .

# Expose the port
EXPOSE 5000

# Run the app
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
