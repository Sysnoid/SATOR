---
title: Local HTTPS Setup
sidebar_position: 11
slug: /local-https-setup
---

# 11. Local HTTPS Setup

This chapter walks through generating locally trusted TLS certificates and
serving SATOR over HTTPS from your development machine. For production TLS,
see [§10.5.2 Reverse-proxy TLS](10-operations.md#1052-reverse-proxy-tls-recommended).

---

## 11.1 Install `mkcert`

`mkcert` generates certificates signed by a locally trusted root CA, so
browsers and HTTP clients accept them without warnings.

### Windows

Pick one:

```powershell
# Winget
winget install --id FiloSottile.mkcert -e
mkcert -install
```

```powershell
# Chocolatey
choco install mkcert -y
mkcert -install
```

```powershell
# Scoop
scoop bucket add extras
scoop install mkcert
mkcert -install
```

> **Important.** After installing, **close the current PowerShell window** and
> open a new one before running further `mkcert` commands. Windows only
> applies `PATH` changes to newly spawned shells.

If `mkcert` is still not found, pull the Winget Links directory into your
current session:

```powershell
$links = "$env:LOCALAPPDATA\Microsoft\WinGet\Links"
if (Test-Path $links) { $env:Path += ";$links" }
```

### macOS

```bash
brew install mkcert nss
mkcert -install
```

### Linux

Install `mkcert` from your distribution (or grab the binary release), then:

```bash
mkcert -install
```

You may need `nss` and `ca-certificates` packages for browser trust.

## 11.2 Generate localhost certificates

From the repository root:

```powershell
mkdir certs
mkcert -cert-file certs\localhost.pem -key-file certs\localhost-key.pem localhost 127.0.0.1 ::1
```

This produces two files:

| File | Purpose |
|---|---|
| `certs\localhost.pem`      | Certificate (chain + leaf). |
| `certs\localhost-key.pem`  | Private key. |

`.gitignore` already excludes `certs/` so your key is never committed.

## 11.3 Enable HTTPS in the server

Add these lines to `.env` (adjust paths to match the files you produced):

```dotenv
SATOR_ENABLE_TLS=true
SATOR_HTTP_PORT=8443
SATOR_TLS_CERT_FILE=certs/localhost.pem
SATOR_TLS_KEY_FILE=certs/localhost-key.pem
# optional:
# SATOR_TLS_KEY_PASSWORD=...
# SATOR_TLS_CA_CERTS=certs/ca-bundle.pem
```

Start the server as usual:

```powershell
sator-server
```

The server will log that it is listening on `https://0.0.0.0:8443`.

## 11.4 Verify from a client

```bash
curl -s https://localhost:8443/livez
# → {"status":"ok"}
```

In a browser, `https://localhost:8443/livez` should display the JSON payload
without a certificate warning.

## 11.5 Alternative: Office.js dev certificates

If you are developing an Office add-in, Microsoft ships its own certificate
helper:

```powershell
npx office-addin-dev-certs install
```

Point `SATOR_TLS_CERT_FILE` and `SATOR_TLS_KEY_FILE` at the files it produces
and follow §11.3.

## 11.6 Troubleshooting

- **`mkcert` not found after install** — reopen PowerShell, or append both
  user and machine `PATH`s in your current shell:
  ```powershell
  $user    = [Environment]::GetEnvironmentVariable('Path','User')
  $machine = [Environment]::GetEnvironmentVariable('Path','Machine')
  $env:Path = "$user;$machine"
  $links = "$env:LOCALAPPDATA\Microsoft\WinGet\Links"
  if (Test-Path $links) { $env:Path += ";$links" }
  ```
- **`FileNotFoundError` at startup** — verify `SATOR_TLS_CERT_FILE` and
  `SATOR_TLS_KEY_FILE` point to existing files. Use absolute paths if
  working-directory assumptions bite you.
- **Browser still warns** — run `mkcert -install` again (as administrator on
  Windows, or with `sudo` on Linux) to install the local CA into the
  system trust store.

## 11.7 Production

Do **not** use `mkcert` certificates in production; they are only trusted on
hosts where the mkcert root CA is installed. In production, terminate TLS at
a reverse proxy with certificates from a public CA (e.g. Let’s Encrypt via
Caddy or Certbot). See
[§10.5.2 Reverse-proxy TLS](10-operations.md#1052-reverse-proxy-tls-recommended).
