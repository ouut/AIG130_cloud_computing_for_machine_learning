# ─────────────────────────────────────────────
#  Dockerfile - Tells Cloud Run how to run the API
#  You do NOT need to understand this deeply.
#  Just make sure this file is in the same folder as main.py
# ─────────────────────────────────────────────

# Start from a Python 3.11 base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements first (for faster rebuilds)
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Tell Cloud Run which port our app listens on
ENV PORT=8080
EXPOSE 8080

# Command to start the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
