# ReMe Memory API Deployment

## 1) Prepare DNS + EC2
1. Point your domain/subdomain (for example `api.example.com`) to your EC2 public IP.
2. Open inbound security group rules for ports `80` and `443`.
3. Install Docker and Docker Compose plugin on EC2.

## 2) Configure environment
1. Copy `.env.deploy.example` to `.env`.
2. Set `DOMAIN` and `ACME_EMAIL`.
3. Set memory backend env vars (`AGENT_MEMORY_DB_URI`, API keys, etc).

Note: `DOMAIN` and `ACME_EMAIL` are read by Compose for Caddy. Keep them in shell env or `.env` in project root.

## 3) Start services
```bash
docker compose up -d --build
```

## 4) Verify
```bash
curl -s https://$DOMAIN/health
```

## API routes
- `POST /memories`
- `POST /memories/query`
- `POST /memories/short-lived`
- `POST /memories/short-lived/query`
- `GET /health`

## Notes
- Caddy auto-manages TLS certificates via Let's Encrypt.
- Certificate issuance requires public DNS resolution and reachable ports `80/443`.
- Persisted certs/config are stored in Docker volumes: `caddy_data`, `caddy_config`.
