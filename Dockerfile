# Multi-stage build: llama-swap with Svelte UI and Prometheus /metrics endpoint

# Stage 1: Build Svelte UI
FROM node:22-alpine AS ui-builder

WORKDIR /build/ui-svelte
COPY ui-svelte/package.json ui-svelte/package-lock.json ./
COPY ui-svelte/.npmrc ./.npmrc
RUN npm ci --legacy-peer-deps
COPY ui-svelte/ .
RUN npm run build
# vite outputs to ../proxy/ui_dist -> /build/proxy/ui_dist

# Stage 2: Build Go binary
FROM golang:1.24-alpine AS go-builder

WORKDIR /build
COPY go.mod go.sum ./
RUN go mod download
COPY . .

# Overlay built UI into proxy/ui_dist
RUN rm -rf proxy/ui_dist
COPY --from=ui-builder /build/proxy/ui_dist ./proxy/ui_dist/

RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-s -w" -o /app/llama-swap .

# Stage 3: Runtime
FROM alpine:3.21

RUN apk --no-cache add ca-certificates

WORKDIR /app
COPY --from=go-builder /app/llama-swap .
COPY --from=go-builder /build/config.example.yaml ./config.yaml
COPY --from=go-builder /build/docs/examples/grafana-dashboard.json ./grafana-dashboard.json

RUN addgroup -g 10001 appgroup && \
    adduser -u 10001 -G appgroup -D appuser && \
    chown -R appuser:appgroup /app

USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/ || exit 1

ENTRYPOINT ["/app/llama-swap"]
CMD ["-config", "/app/config.yaml", "-listen", "0.0.0.0:8080"]
