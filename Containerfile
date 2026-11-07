# ==============================================================================
# Containerfile (Especificación OCI compatible con Podman y Docker)
# Proyecto: MiBici MLOps - Servicio de Inferencia API REST con FastAPI
# ==============================================================================

# 1. Imagen base oficial de Python 3.10 slim (reproducible, ligera y segura)
FROM python:3.10-slim

# Metadatos del contenedor bajo el estándar Open Container Initiative (OCI)
LABEL org.opencontainers.image.title="MiBici MLOps Inference API" \
      org.opencontainers.image.description="API REST con FastAPI para servir predicciones del modelo Champion de MiBici (CatBoost P1)" \
      org.opencontainers.image.authors="Equipo 51" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.licenses="MIT"

# 2. Variables de entorno estándar de Python y configuración de red/MLflow
# - PYTHONDONTWRITEBYTECODE: evita crear archivos .pyc dentro del contenedor
# - PYTHONUNBUFFERED: asegura que los logs de Uvicorn/FastAPI salgan inmediatamente a stdout/stderr
# - MLFLOW_TRACKING_URI: configurable en tiempo de ejecución (por defecto apunta al host local)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    MLFLOW_TRACKING_URI="http://host.containers.internal:5000"

# 3. Establecer directorio de trabajo en la ruta estándar /app
WORKDIR /app

# 4. Crear usuario no privilegiado por seguridad (evita ejecución como root en el contenedor)
RUN useradd -m -u 1000 -s /bin/bash appuser

# 5. Instalar dependencias de terceros antes de copiar el código fuente.
# Se excluye "-e ." porque el proyecto local aún no está disponible.
COPY requirements.txt ./

RUN pip install --no-cache-dir --upgrade pip && \
    sed '/^[[:space:]]*-e[[:space:]]*\.[[:space:]]*$/d' requirements.txt \
    > /tmp/requirements-docker.txt && \
    pip install --no-cache-dir -r /tmp/requirements-docker.txt && \
    rm /tmp/requirements-docker.txt

# 6. Copiar e instalar el paquete local MiBici.
# pyproject.toml requiere README.md y LICENSE.
COPY pyproject.toml README.md LICENSE ./
COPY mibici_mlops ./mibici_mlops

RUN pip install --no-cache-dir --no-deps -e .

# 7. Asignar propiedad de los archivos al usuario no privilegiado y cambiar de contexto
RUN chown -R appuser:appuser /app
USER appuser

# 8. Exponer el puerto del servicio HTTP
EXPOSE 8000

# 9. Healthcheck para monitoreo nativo del estado del contenedor en Podman / Docker
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# 10. Comando de arranque: servidor ASGI Uvicorn en modo producción escuchando en 0.0.0.0
CMD ["uvicorn", "mibici_mlops.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
