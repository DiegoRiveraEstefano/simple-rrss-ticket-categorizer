# ==========================================
# Stage 1: Builder - instala dependencias
# ==========================================
FROM python:3.12-slim AS builder

# Instalar solo herramientas necesarias para compilar deps nativas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Crear entorno virtual
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Instalar dependencias de Python (sin caché, sin pip cache)
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt


# ==========================================
# Stage 2: Runtime - imagen final mínima
# ==========================================
FROM python:3.12-slim AS runtime

# Copiar entorno virtual y datos NLTK ya preparados
COPY --from=builder /opt/venv /opt/venv

# Configurar entorno
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app

# Copiar solo el código necesario
COPY app/ ./app
COPY models/ ./models
COPY simple_rrss_ticket_categorizer/ ./simple_rrss_ticket_categorizer


# 4. Exponer el puerto y ejecutar la aplicación
EXPOSE 8000
CMD ["/opt/venv/bin/python", "-m", "app.main"]

