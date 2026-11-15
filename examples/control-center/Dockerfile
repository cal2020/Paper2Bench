# Stage this image explicitly before a network-disabled Control Center run.
# Building downloads Python dependencies; verification itself installs nothing.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /opt/paper2bench-build
COPY pyproject.toml README.md LICENSE ./
COPY paper2bench ./paper2bench
RUN python -m pip install --no-cache-dir '.[repro]'
WORKDIR /workspace
