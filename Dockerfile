# 1 Base image /use any version of this python
FROM python:3.10-slim

#working directory  #Creates a folder inside container called /app  Everything will run from there.
WORKDIR /app   

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*


COPY requirements-prod.txt .

RUN pip install --no-cache-dir -r requirements-prod.txt

#copy entire project
COPY . .

#Declares that FastAPI will run on port 8000.
EXPOSE 8000

#run api telling docker how to start api
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
 