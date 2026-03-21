FROM node:22-slim

# System deps
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    git curl bash procps jq \
    && rm -rf /var/lib/apt/lists/*

# Pi coding agent
RUN npm install -g @mariozechner/pi-coding-agent@0.54.2

# LOS CLI — installed globally so models can use `los` command
COPY open-los/packages/cli /tmp/los-cli
RUN cd /tmp/los-cli && npm install && npm run build && npm link && rm -rf /tmp/los-cli

# Workspace
WORKDIR /workspace

# LOS docs — copied in at build time (TASK.md overridden at runtime via -v mount)
COPY docker/agent/TASK.md /workspace/TASK.md
COPY open-los/docs/AGENTS.md /workspace/LOS_DOCS.md
COPY open-los/openapi/v1.yaml /workspace/openapi.yaml

# Pi config
RUN mkdir -p /root/.pi/agent

ENTRYPOINT ["pi", "--mode", "rpc", "--no-session", "--provider", "openrouter"]
