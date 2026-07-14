FROM python:3.11-slim

WORKDIR /app

# Instala dependencias primero (aprovecha cache de Docker si el codigo cambia pero no requirements.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del backend
COPY . .

# Render inyecta la variable $PORT en tiempo de ejecucion
EXPOSE 8000
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
