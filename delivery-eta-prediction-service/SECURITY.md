# Security Setup Guide

## Critical: First-Time Setup

**DO NOT USE DEFAULT CREDENTIALS IN PRODUCTION!**

This guide helps you securely configure the delivery ETA prediction service.

---

## Quick Setup (Development)

### 1. Generate Secure Credentials

```bash
# Generate a Fernet key for Airflow
python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# Generate random passwords (example using openssl)
openssl rand -base64 32
```

### 2. Create Environment File

Copy the example file and update all values:

```bash
cp .env.example .env
```

Edit `.env` and replace ALL placeholder values:

```env
# PostgreSQL Database Credentials
POSTGRES_USER=airflow_admin              # Change this
POSTGRES_PASSWORD=<generated-password>   # Use generated password
POSTGRES_DB=airflow

# Airflow Fernet Key
AIRFLOW__CORE__FERNET_KEY=<generated-fernet-key>  # Use generated key

# Airflow Admin Credentials
AIRFLOW_ADMIN_USERNAME=<your-username>     # Change this
AIRFLOW_ADMIN_PASSWORD=<generated-password> # Use generated password

# Grafana Admin Password
GRAFANA_ADMIN_PASSWORD=<generated-password> # Use generated password

# API CORS Configuration
ALLOWED_ORIGINS=http://localhost:3001,http://localhost:3000
```

### 3. Create Frontend Environment File

```bash
cp frontend/.env.example frontend/.env.local
```

Get your Mapbox token from https://account.mapbox.com/access-tokens/ and update:

```env
NEXT_PUBLIC_MAPBOX_TOKEN=pk.your_actual_token_here
```

### 4. Load Environment Variables

```bash
# Export all environment variables
export $(cat .env | xargs)

# Verify critical variables are set
echo $AIRFLOW__CORE__FERNET_KEY
echo $POSTGRES_PASSWORD
```

### 5. Start Services

```bash
docker-compose up -d
```

---

## Production Deployment

### Required Actions Before Production:

1. **Rotate All Credentials**
   - Generate new, strong passwords for all services
   - Use a password manager to store credentials securely
   - Never reuse development credentials in production

2. **Use Secrets Management**
   - AWS Secrets Manager / Azure Key Vault / HashiCorp Vault
   - Kubernetes Secrets (if deploying to K8s)
   - Docker Secrets (for Docker Swarm)

3. **Configure CORS Properly**
   Update `ALLOWED_ORIGINS` to your actual domain(s):
   ```env
   ALLOWED_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
   ```

4. **Enable HTTPS**
   - Use a reverse proxy (nginx, Traefik, Caddy)
   - Obtain SSL certificates (Let's Encrypt, etc.)
   - Redirect all HTTP traffic to HTTPS

5. **Database Security**
   - Use managed database services in production
   - Enable SSL/TLS for database connections
   - Implement connection pooling
   - Regular backups with encryption

6. **Network Security**
   - Place services behind a VPN or private network
   - Use firewall rules to restrict access
   - Implement rate limiting
   - Enable audit logging

---

## Security Checklist

### Before First Commit
- [ ] All credentials moved to `.env` file
- [ ] `.env` is in `.gitignore`
- [ ] `.env.example` has placeholder values only
- [ ] No hardcoded secrets in code
- [ ] No API keys in frontend code (except public tokens in `.env.local`)

### Before Production Deploy
- [ ] All credentials rotated from development
- [ ] Fernet key regenerated
- [ ] Database uses strong password (20+ characters)
- [ ] Admin passwords are unique and complex
- [ ] CORS configured with specific domain(s)
- [ ] HTTPS enabled with valid certificates
- [ ] Secrets stored in secure vault
- [ ] Database backups configured
- [ ] Monitoring and alerting enabled
- [ ] Rate limiting implemented
- [ ] Firewall rules configured

---

## Credential Rotation Schedule

Rotate credentials regularly:

| Credential | Frequency | Priority |
|------------|-----------|----------|
| Fernet Key | Every 90 days | Critical |
| Database Password | Every 90 days | Critical |
| Admin Passwords | Every 60 days | High |
| API Tokens | Every 30 days | Medium |

---

## Emergency Response

### If Credentials Are Exposed:

1. **Immediately rotate all affected credentials**
2. **Check git history for exposed secrets**
   ```bash
   git log -p | grep -i "password\|key\|token"
   ```
3. **Remove secrets from git history** (if exposed)
   ```bash
   # Use git-filter-repo or BFG Repo Cleaner
   pip install git-filter-repo
   git filter-repo --path-glob '**/.env' --invert-paths
   ```
4. **Revoke compromised API tokens** (Mapbox, etc.)
5. **Review access logs** for suspicious activity
6. **Notify your security team**

### Clean Git History (if secrets were committed)

```bash
# Install BFG Repo Cleaner
brew install bfg  # macOS
# or download from: https://rtyley.github.io/bfg-repo-cleaner/

# Remove sensitive files from history
bfg --delete-files .env
bfg --delete-files .env.local

# Clean up and force push
git reflog expire --expire=now --all
git gc --prune=now --aggressive
git push --force --all
```

**Warning:** Only force push if you're certain no one else has pulled the compromised commits.

---

## Additional Resources

- [OWASP Secrets Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)
- [Airflow Security Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/security/index.html)
- [FastAPI Security Guide](https://fastapi.tiangolo.com/tutorial/security/)

---

## Contact

For security issues or questions, please contact your security team or project maintainer.

**Never share credentials via:**
- Email
- Slack/Discord
- Code repositories
- Screenshots
- Issue trackers
