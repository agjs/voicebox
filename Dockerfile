FROM python:3.11-slim

ENV HF_HOME=/models/huggingface
# Cap OpenMP/MKL to the default container CPU quota before native libs load.
# Override at runtime via compose (OMP_NUM_THREADS/MKL_NUM_THREADS from VOICEBOX_CPU_THREADS).
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

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
# Inside a container the process must listen on all interfaces; host publish
# (compose VOICEBOX_BIND_ADDRESS) controls external exposure.
ENV VOICEBOX_BIND_ADDRESS=0.0.0.0
ENV VOICEBOX_ALLOW_INSECURE_BIND=true
EXPOSE 8790
CMD ["python", "-m", "voicebox"]
