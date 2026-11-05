# ==============================================================================
# Containerfile (Compatible con estándar OCI para Podman y Docker)
# Proyecto: MiBici MLOps - Servicio de Inferencia API REST con FastAPI
# Responsable: Ingrid
# ==============================================================================

# TODO - Ingrid - Definir imagen base oficial de Python 3.10 slim (reproducible y ligera)
FROM python:3.10-slim

# TODO - Ingrid - Configurar variables de entorno para Python y MLflow dentro del contenedor
# PYTHONDONTWRITEBYTECODE=1, PYTHONUNBUFFERED=1, PORT=8000
# MLFLOW_TRACKING_URI configurable en tiempo de ejecución (ej. http://host.containers.internal:5000)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    MLFLOW_TRACKING_URI="http://host.containers.internal:5000"

# TODO - Ingrid - Establecer directorio de trabajo en /app
WORKDIR /app

# TODO - Ingrid - Crear usuario no privilegiado por seguridad (cumplimiento OCI/seguridad)
# RUN useradd -m -u 1000 appuser

# TODO - Ingrid - Copiar e instalar dependencias primero para optimizar el caché de capas
# COPY requirements.txt pyproject.toml ./
# RUN pip install --no-cache-dir -r requirements.txt

# TODO - Ingrid - Copiar código fuente del proyecto y cambiar permisos a appuser
# COPY mibici_mlops ./mibici_mlops
# RUN chown -R appuser:appuser /app
# USER appuser

# TODO - Ingrid - Exponer puerto 8000
EXPOSE 8000

# TODO - Ingrid - Definir comando de arranque con Uvicorn escuchando en 0.0.0.0
# CMD ["uvicorn", "mibici_mlops.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

