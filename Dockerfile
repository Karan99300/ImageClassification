FROM python:3.11-slim

EXPOSE 5002

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /app
COPY . /app

# Set the Gunicorn timeout to 120 seconds
CMD ["gunicorn", "--bind", "0.0.0.0:5002", "--timeout", "120", "app:app"]
