# VeriFile-X — Deployment Guide

This guide covers every way to run VeriFile-X: personal laptop, Docker, cloud, organisation server, government on-premises, and the live public demo. Each section is written for complete beginners — no prior server experience needed.

---

## Table of Contents

1. [Requirements](#1-requirements)
2. [Local Setup — Personal Use](#2-local-setup--personal-use)
3. [Docker — Any Machine, One Command](#3-docker--any-machine-one-command)
4. [Cloud Deployment](#4-cloud-deployment)
5. [Organisation Deployment](#5-organisation-deployment)
6. [Government / On-Premises Deployment](#6-government--on-premises-deployment)
7. [Online Demo (No Install)](#7-online-demo-no-install)
8. [Environment Variables](#8-environment-variables)
9. [Troubleshooting](#9-troubleshooting)
10. [Security Checklist](#10-security-checklist)

---

## 1. Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10 | 3.11 |
| RAM | 4 GB | 8 GB |
| Disk | 5 GB free | 20 GB free |
| OS | Windows 10, macOS 12, Ubuntu 20.04 | Ubuntu 22.04 LTS |

---

## 2. Local Setup — Personal Use

### Step 1 — Install Python

Download from [python.org/downloads](https://www.python.org/downloads). On Windows, check **Add Python to PATH** during install.

```bash
python --version    # should print 3.10 or newer
```

### Step 2 — Get the code

```bash
git clone https://github.com/abinaze/VeriFile-X.git
cd VeriFile-X
```

No git? Download the ZIP from GitHub and unzip it.

### Step 3 — Create a virtual environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You will see `(venv)` in your prompt when it is active.

### Step 4 — Install dependencies

```bash
pip install -r backend/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu
```

This takes 3–10 minutes. On Linux you may also need:

```bash
sudo apt-get install libmagic1 libmagic-dev file
```

On macOS:
```bash
brew install libmagic
```

### Step 5 — Configure (optional)

```bash
cp .env.example .env
# Edit .env if you want to change any settings
```

### Step 6 — Start the server

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
INFO: Uvicorn running on http://0.0.0.0:8000
```

### Step 7 — Open the frontend

Open `frontend/index.html` in your browser. It connects automatically to the server.

Or go to `http://localhost:8000/docs` for the interactive API documentation.

Press `Ctrl+C` to stop the server.

---

## 3. Docker — Any Machine, One Command

Docker packages everything so you do not need to install Python or any libraries manually. It works identically on Windows, macOS, and Linux.

### Step 1 — Install Docker

Download [Docker Desktop](https://www.docker.com/products/docker-desktop) and follow the instructions for your OS.

```bash
docker --version    # should print Docker version 24.x or newer
```

### Step 2 — Get the code and configure

```bash
git clone https://github.com/abinaze/VeriFile-X.git
cd VeriFile-X
cp .env.example .env    # edit if needed
```

### Step 3 — Build and run

```bash
docker build -t verifile-x .
docker run --env-file .env -p 8000:7860 verifile-x
```

Open `http://localhost:8000` in your browser.

### Run in background

```bash
docker run -d --env-file .env -p 8000:7860 --name vfx --restart unless-stopped verifile-x
```

### Useful Docker commands

```bash
docker logs vfx          # view server logs
docker stop vfx          # stop the container
docker start vfx         # start it again
docker rm vfx            # remove it completely
```

---

## 4. Cloud Deployment

### Option A — Hugging Face Spaces (free, recommended)

VeriFile-X is pre-configured for Hugging Face Spaces.

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Click **New Space** → choose **Docker** SDK
3. Connect your GitHub repository or upload the files
4. The space builds automatically — your API is live at `https://your-username-verifile-x-api.hf.space`

The frontend can be served for free on GitHub Pages:
1. Go to your repository → **Settings** → **Pages**
2. Source: `main` branch, `/ (root)` folder
3. GitHub deploys automatically on every push

### Option B — Any cloud virtual machine

All major providers work. Use Ubuntu 22.04 LTS.

```bash
# Connect to your VM
ssh user@your-server-ip

# Install Docker
curl -fsSL https://get.docker.com | bash
sudo usermod -aG docker $USER
# Log out and back in

# Deploy VeriFile-X
git clone https://github.com/abinaze/VeriFile-X.git
cd VeriFile-X
cp .env.example .env
docker build -t verifile-x .
docker run -d --env-file .env -p 80:7860 --name vfx --restart unless-stopped verifile-x
```

Add HTTPS:
```bash
sudo apt install nginx certbot python3-certbot-nginx -y
sudo certbot --nginx -d your-domain.com
```

---

## 5. Organisation Deployment

For a team of analysts sharing one server.

### Architecture

```
[Analysts / Browsers]
        ↓
[Nginx — HTTPS, rate limiting]
        ↓
[VeriFile-X API — Docker]
        ↓
[Local data/ directory — cases, keys, audit log]
```

### Step-by-step

**1. Provision a server**

- 4–8 vCPUs, 8–16 GB RAM, 50 GB SSD
- Ubuntu 22.04 LTS
- A domain name

**2. Install dependencies**

```bash
sudo apt update
sudo apt install -y docker.io docker-compose nginx certbot python3-certbot-nginx
sudo systemctl enable docker
sudo usermod -aG docker $USER
```

**3. Deploy VeriFile-X**

```bash
git clone https://github.com/abinaze/VeriFile-X.git /opt/verifile-x
cd /opt/verifile-x
cp .env.example .env
# Edit .env — set ADMIN_KEY_HASH (see Security section below)
docker build -t verifile-x .
docker run -d \
  --env-file .env \
  -p 127.0.0.1:8000:7860 \
  -v /opt/verifile-x/data:/app/data \
  --name vfx \
  --restart unless-stopped \
  verifile-x
```

**4. Configure Nginx**

Create `/etc/nginx/sites-available/verifile-x`:

```nginx
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-Proto https;
        client_max_body_size 15M;
    }

    location / {
        root /opt/verifile-x/frontend;
        try_files $uri /index.html;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/verifile-x /etc/nginx/sites-enabled/
sudo certbot --nginx -d your-domain.com
sudo systemctl restart nginx
```

**5. Create API keys for your team**

Generate the admin key hash and put it in `.env`:

```bash
python3 -c "import hashlib,secrets; k=f'vfx_{secrets.token_urlsafe(32)}'; print('Key:',k); print('Hash:',hashlib.sha256(k.encode()).hexdigest())"
# Copy the Hash value into .env as ADMIN_KEY_HASH=<hash>
# Save the Key value — share it only with admins
```

Create analyst keys through the API:
```bash
curl -X POST https://your-domain.com/api/v1/keys/ \
  -H "X-Admin-Key: your-admin-key" \
  -H "Content-Type: application/json" \
  -d '{"name": "Analyst Name", "role": "analyst"}'
```

Roles:

| Role | Access |
|------|--------|
| `admin` | Everything including key management |
| `analyst` | All analysis and case endpoints |
| `viewer` | Read-only access to results |

**6. Automated backups**

```bash
# Daily backup at 2am
echo "0 2 * * * tar -czf /backup/vfx-\$(date +\%Y\%m\%d).tar.gz /opt/verifile-x/data/" | crontab -
```

---

## 6. Government / On-Premises Deployment

For regulated environments requiring network isolation, full audit trails, and no external dependencies.

### Key requirements

- No outbound internet required after initial setup (models load from local disk)
- All data stored on your own server
- Audit log is append-only with SHA-256 hash chaining — tamper-evident
- API keys stored as SHA-256 hashes — raw keys never on disk

### Step-by-step

**1. Provision**

- 8+ vCPUs, 16+ GB RAM, 200 GB SSD (encrypted)
- Ubuntu 22.04 LTS with latest security patches
- Firewall: allow only ports 443 and 22 externally

```bash
sudo ufw allow 22
sudo ufw allow 443
sudo ufw enable
```

**2. Transfer files (air-gapped)**

If the server has no internet:
```bash
# On a connected machine:
git clone https://github.com/abinaze/VeriFile-X.git
tar -czf verifile-x.tar.gz VeriFile-X/
# Transfer to the server, then:
tar -xzf verifile-x.tar.gz
cd VeriFile-X
```

**3. Network-isolated Docker**

```bash
docker network create --internal vfx-internal

docker run -d \
  --network vfx-internal \
  --env-file .env \
  -p 127.0.0.1:8000:7860 \
  -v $(pwd)/data:/app/data \
  --name vfx \
  --restart unless-stopped \
  verifile-x
```

This prevents any outbound connections from the container.

**4. Nginx with IP allowlist**

```nginx
server {
    listen 443 ssl;
    server_name verifile-x.internal.example.gov;

    allow 10.0.0.0/8;
    allow 192.168.0.0/16;
    deny all;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header X-Forwarded-Proto https;
        client_max_body_size 15M;
    }
}
```

**5. Audit log**

VeriFile-X writes all analysis events to `data/audit_log.jsonl`. Each entry includes:
- SHA-256 hash of the analyzed file (not the file itself)
- Analysis verdict and signal scores
- Timestamp and analyzer version
- Hash of the previous entry (tamper-evident chain)

Forward to your SIEM:
```bash
tail -F data/audit_log.jsonl | logger -t verifile-x
```

**6. Notes for compliance**

- VeriFile-X does not retain uploaded images after analysis
- Analysis results are stored only in the in-memory cache (cleared on restart) and the audit log
- No telemetry or analytics calls are made

---

## 7. Online Demo (No Install)

Use VeriFile-X right now without installing anything:

- **Web interface:** [abinaze.github.io/VeriFile-X](https://abinaze.github.io/VeriFile-X)
- **API:** [abinazebinoy-verifile-x-api.hf.space](https://abinazebinoy-verifile-x-api.hf.space)
- **Interactive docs:** [abinazebinoy-verifile-x-api.hf.space/docs](https://abinazebinoy-verifile-x-api.hf.space/docs)

---

## 8. Environment Variables

Create a `.env` file in the project root to override defaults.

| Variable | Default | Description |
|----------|---------|-------------|
| `CORS_ORIGINS` | localhost + production URLs | Comma-separated allowed CORS origins |
| `DEBUG` | `False` | Enable debug mode |
| `RATE_LIMIT_PER_MINUTE` | `10` | Requests per minute per IP |
| `MAX_FILE_SIZE_MB` | `50` | Upload validation limit |
| `MAX_ANALYSIS_SIZE_MB` | `10` | Analysis processing limit |
| `CACHE_TTL_MINUTES` | `60` | How long to cache analysis results |
| `MAX_CACHE_SIZE` | `500` | Maximum number of cached results |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`) |
| `ADMIN_KEY_HASH` | *(unset)* | SHA-256 hash of the admin key |

**Example `.env` for production:**
```env
CORS_ORIGINS=https://your-domain.com
DEBUG=False
RATE_LIMIT_PER_MINUTE=20
MAX_ANALYSIS_SIZE_MB=10
ADMIN_KEY_HASH=<sha256-hash-of-your-admin-key>
LOG_LEVEL=WARNING
```

---

## 9. Troubleshooting

| Problem | Check |
|---------|-------|
| Server won't start | Python ≥ 3.10? Virtual env active? Run `pip install -r backend/requirements.txt` again |
| `libmagic` error | Linux: `sudo apt install libmagic1` · macOS: `brew install libmagic` |
| Docker build fails | Ensure Docker Desktop is running and you have 5+ GB disk space |
| Frontend shows blank page | Backend running? Visit `http://localhost:8000/health` |
| CORS error in browser | Add your origin to `CORS_ORIGINS` in `.env` |
| 415 Unsupported Media Type | Only JPEG, PNG, WebP are accepted |
| 413 Payload Too Large | File exceeds 10 MB analysis limit |
| Metrics reset returns 401 | Set `ADMIN_KEY_HASH` in `.env` and send the matching key |
| Analysis is slow | First request loads models from disk — subsequent requests use cache |
| Container exits immediately | Check `docker logs vfx` for the error |

---

## 10. Security Checklist

Before exposing VeriFile-X to any external access:

- [ ] HTTPS enabled with valid SSL certificate
- [ ] HTTP → HTTPS redirect configured
- [ ] `ADMIN_KEY_HASH` set in environment — not just key length
- [ ] Only ports 443 and 22 open externally
- [ ] `DEBUG=False` in production
- [ ] Admin key distributed through a secure channel
- [ ] Daily backups configured and tested
- [ ] Audit log forwarding to log management system
- [ ] `/health` endpoint monitored by external uptime checker
- [ ] Rate limits appropriate for your user count
