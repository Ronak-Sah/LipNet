# ---- Base image ----
FROM python:3.10-slim

# ---- Environment settings ----
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ---- System dependencies ----
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---- Working directory ----
WORKDIR /app

# ---- Copy requirements ----
COPY requirements.txt .

# ---- Install Python dependencies ----
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# ---- Copy project files ----
COPY . .

# ---- Expose API port ----
EXPOSE 8000

# ---- Run FastAPI ----
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
