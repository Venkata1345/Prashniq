# Stage 1: build the React frontend
FROM node:22-alpine AS frontend
WORKDIR /fe
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: the Python app
FROM python:3.11-slim
WORKDIR /srv

# Install dependencies first so code edits don't bust the pip cache layer.
COPY pyproject.toml ./
COPY app ./app
RUN pip install --no-cache-dir .

COPY migrations ./migrations
COPY alembic.ini serve.py ./
COPY app/static ./app/static
COPY --from=frontend /fe/dist ./frontend/dist

ENV PYTHONUNBUFFERED=1 HOST=0.0.0.0
EXPOSE 8000
CMD ["python", "serve.py"]
