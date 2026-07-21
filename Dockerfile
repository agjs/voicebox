FROM python:3.11-slim

ENV HF_HOME=/models/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg espeak-ng espeak curl && rm -rf /var/lib/apt/lists/*

RUN groupadd --system voicebox && useradd --system --gid voicebox --home-dir /app voicebox \
    && mkdir -p "$HF_HOME" && chown -R voicebox:voicebox /models

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --disable-pip-version-check -e .

# Bake models into the image at build time (reproducible, offline runtime).
COPY scripts/fetch_models.py ./scripts/fetch_models.py
USER voicebox
RUN python scripts/fetch_models.py

# After models are cached, ensure runtime reads only from the cache
ENV HF_HUB_OFFLINE=1

ENV VOICEBOX_PORT=8790
EXPOSE 8790
CMD ["python", "-m", "voicebox"]
