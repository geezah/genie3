FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.12 /uv /uvx /bin/

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    apt-get install -y --no-install-recommends build-essential gcc supervisor nginx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY packages/federated/server_config/supervisord.conf /supervisord.conf
COPY packages/federated/server_config/nginx /etc/nginx/sites-available/default
COPY packages/federated/server_config/docker-entrypoint.sh /entrypoint.sh

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-workspace --all-packages

ADD . /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --all-packages

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 9000 9001
ENTRYPOINT ["sh", "/entrypoint.sh"]
