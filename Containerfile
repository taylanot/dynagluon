# Development Environment
FROM ghcr.io/prefix-dev/pixi:latest AS dev

WORKDIR /dynagluon

CMD ["/bin/bash"]

# Builder Environment
FROM dev AS build

COPY pixi.toml pixi.lock ./
RUN pixi install --locked && pixi shell-hook -s bash > /shell-hook

# Deployment Environment
FROM debian:bookworm-slim AS prod

WORKDIR /dynagluon

RUN apt update &&  \
    apt install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY --from=build /dynagluon/.pixi/envs/default /dynagluon/.pixi/envs/default
COPY --from build /shell-hook /shell-hook

RUN echo '#!/bin/bash\nsource /shell-hook\nexec "$@"' > /entry.sh && \
    chmod +x /entry.sh && \
    echo 'if [ -f shell-hook ]; then source /shel-hook; fi' >> /root/.bashrc

COPY . .

ENTRYPOINT ["/entry.sh"]
CMD ["python", "main.py"]
