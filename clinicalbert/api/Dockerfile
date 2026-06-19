# Base image: minimal Python, no unnecessary OS bloat
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy only requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the actual project code
COPY clinicalbert/ ./clinicalbert/

# Expose the port FastAPI/uvicorn will run on
EXPOSE 8000

# Command to run when the container starts
CMD ["uvicorn", "clinicalbert.api.main:app", "--host", "0.0.0.0", "--port", "8000"]