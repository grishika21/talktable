# Running TalkTable with Docker

`docker-compose.yml` starts everything TalkTable needs, all on your machine:

| Service | What it does |
|---|---|
| `postgres`, `mongo` | The sales and engagement databases. Their read-only users are created on first start. They have no published ports, so only the app can reach them. |
| `seed` | Fills both databases with the demo data, then exits. The data is deterministic, so re-running it is harmless. |
| `ollama` | The local LLM server. |
| `ollama-pull` | Downloads the models into the `ollama` volume once, then exits. |
| `app` | TalkTable: chat UI and MCP server on http://127.0.0.1:7860 |

Needs Docker with the Compose plugin (`docker compose version`). The SQLite book catalog
ships inside the image.

## Start

```bash
docker compose up --build
```

The first start downloads the models, about 14 GB for the defaults (qwen3:14b for routing and
answers, qwen2.5-coder:7b for writing queries), so it takes a while. Later starts reuse them.
Then open **http://127.0.0.1:7860**. MCP clients connect to
`http://127.0.0.1:7860/gradio_api/mcp`.

Stop with `Ctrl+C`, or `docker compose down` if you started with `-d`. Add `-v` to also delete
the volumes (database data, models, answer cache).

## Variations

**NVIDIA GPU** (needs the NVIDIA Container Toolkit), much faster:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

**Use an Ollama you already have installed,** and its downloaded models:

```bash
docker compose -f docker-compose.yml -f docker-compose.host-ollama.yml up --build
```

On Docker Desktop (macOS, Windows) this works as is. It's also the best option on a Mac,
where native Ollama can use the GPU and Ollama inside Docker can't. On Linux, Ollama only
listens on 127.0.0.1 by default, which containers can't reach. Make it listen on all
interfaces with `sudo systemctl edit ollama`:

```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0"
```

then `sudo systemctl restart ollama`, and keep port 11434 closed in your firewall.

**A Qdrant server for the schema vectors,** instead of keeping them in the app's memory:

```bash
docker compose -f docker-compose.yml -f docker-compose.qdrant.yml up --build
```

Worth it for large schemas (hundreds of tables and more): the vectors persist in a volume,
so restarts don't embed anything again, and several app containers can share one index.
For the 30 demo tables the in-memory NumPy store is just as good. Benchmarks are in the
README.

**Smaller or different models:**

```bash
TALKTABLE_LLM_MODEL=qwen2.5-coder:7b TALKTABLE_QUERY_MODEL=qwen2.5-coder:7b docker compose up --build
```

## Settings

All optional. Set them in your shell or in a `.env` file next to `docker-compose.yml`:

| Variable | Default | Meaning |
|---|---|---|
| `TALKTABLE_LLM_MODEL` | `qwen3:14b` | Model for routing, follow-ups and explanations |
| `TALKTABLE_QUERY_MODEL` | `qwen2.5-coder:7b` | Model that writes the SQL / Mongo queries |
| `TALKTABLE_PORT` | `7860` | Port on 127.0.0.1 for the UI and MCP endpoint |
| `TALKTABLE_RO_PASSWORD` | `talktable_ro_dev` | Password of the read-only database users |
| `TALKTABLE_PG_PASSWORD` | `talktable_dev` | Postgres admin password (used only by `seed`) |
| `TALKTABLE_MONGO_ROOT_PASSWORD` | `root_dev` | Mongo root password (used only by `seed`) |
| `TALKTABLE_OLLAMA_URL` | `http://host.docker.internal:11434` | Only with `docker-compose.host-ollama.yml` |

The default passwords are fine on your own machine. Change them before running this
anywhere else.

## What's in the image

`docker/Dockerfile` builds from `python:3.11-slim`. It installs `requirements.txt` (CPU-only
PyTorch) and bakes in the embedding model, so containers start offline and fast. It runs as
an unprivileged user that can write only to `/data`, where the answer cache lives, in the
`appdata` volume. `.env`, `.venv` and the tests are kept out of the image by `.dockerignore`.
