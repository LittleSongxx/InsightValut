FROM node:22.14.0-bookworm-slim

WORKDIR /app

COPY package.json package-lock.json ./
COPY docker/docker-dev-entrypoint.sh /usr/local/bin/docker-dev-entrypoint.sh

RUN chmod +x /usr/local/bin/docker-dev-entrypoint.sh

EXPOSE 3002

CMD ["docker-dev-entrypoint.sh"]
