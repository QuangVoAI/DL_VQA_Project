# Use standard Python image
FROM python:3.9-slim

# Set working directory in container
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire source code and models directory into the container
COPY . .

# Expose port 8000
EXPOSE 8000

# Command to run the server
CMD ["uvicorn", "web.main:app", "--host", "0.0.0.0", "--port", "8000"]