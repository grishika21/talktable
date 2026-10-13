# Deploying TalkTable to a Hugging Face Space

The Space runs `app.py`: the chat UI plus a remote MCP server. A free CPU Space can't
run an LLM or reach your laptop's Docker databases, so the deployed version uses:

| Piece | Local development | Hugging Face Space |
|---|---|---|
| Book catalog (SQLite) | `data/bookstore.db` | the same file, shipped with the code |
| Sales (PostgreSQL) | Docker | **Neon** (free tier) |
| Engagement (MongoDB) | Docker | **MongoDB Atlas** (free M0 cluster) |
| LLM | Ollama | **Groq** (free tier) or **Hugging Face Inference Providers** |

Plan on about 30-45 minutes. Each step ends with a check, so problems show up early.

---

## 1. PostgreSQL on Neon

1. Sign up at [neon.tech](https://neon.tech) and create a project (Postgres 16 or newer,
   region close to you; `AWS US East` is close to Hugging Face).
2. Open **Databases**, then **New database**, and name it `bookstore`.
3. Open **Connect**, choose the `bookstore` database, and copy the connection string. It
   belongs to the owner role (usually `neondb_owner`) and looks like
   `postgresql://neondb_owner:...@ep-xxxx.us-east-1.aws.neon.tech/bookstore?sslmode=require`.
   This is your **seed URL**: it can write, so it is only used once, from your laptop.
4. **Create the read-only user with SQL.** Neon adds roles created in the Console to
   `neon_superuser`, which can do far more than read, so don't create this one in the
   Console. Open the **SQL Editor**, select the `bookstore` database, and run the following
   with your own long random password. Neon rejects weak ones;
   `python -c "import secrets; print(secrets.token_urlsafe(24))"` makes a good one.

   ```sql
   CREATE ROLE talktable_ro WITH LOGIN PASSWORD 'PASTE-A-LONG-RANDOM-PASSWORD';

   GRANT CONNECT ON DATABASE bookstore TO talktable_ro;
   GRANT USAGE ON SCHEMA public TO talktable_ro;
   GRANT SELECT ON ALL TABLES IN SCHEMA public TO talktable_ro;
   -- Tables the seed script creates later (as neondb_owner) will be readable too:
   ALTER DEFAULT PRIVILEGES FOR ROLE neondb_owner IN SCHEMA public GRANT SELECT ON TABLES TO talktable_ro;

   -- Defence in depth: every session of this role is read-only and time-limited.
   ALTER ROLE talktable_ro SET default_transaction_read_only = on;
   ALTER ROLE talktable_ro SET statement_timeout = '5s';
   ```

   If your owner role isn't called `neondb_owner`, use its name in the `ALTER DEFAULT
   PRIVILEGES` line.
5. Your **read-only URL** is the seed URL with the user and password replaced:
   `postgresql://talktable_ro:<password>@ep-xxxx.us-east-1.aws.neon.tech/bookstore?sslmode=require`.
   Use the host *without* `-pooler`, the direct connection.

## 2. MongoDB on Atlas

1. Sign up at [mongodb.com/atlas](https://www.mongodb.com/atlas) and create a **free (M0)**
   cluster.
2. **Database Access**, then **Add New Database User**, twice (password authentication):
   - `talktable_seed` with the built-in role **Read and write to any database**. This
     account is only used for seeding; delete it afterwards if you like.
   - `talktable_ro` with **Specific Privileges**, then `read` on database `bookstore`. This
     is the account the app uses.
3. **Network Access**, then **Add IP Address**, then **Allow access from anywhere**
   (`0.0.0.0/0`). Hugging Face Spaces don't have fixed outgoing IP addresses, so there is
   no narrower rule that works. The read-only user and a strong password are what protect
   the data.
4. **Connect**, then **Drivers**, and copy the SRV string, e.g.
   `mongodb+srv://<user>:<password>@cluster0.abcde.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0`.
   Fill in `talktable_seed` for the **seed URL** and `talktable_ro` for the **read-only URL**.
   If a password contains `@ : / ? # %`, URL-encode it, or pick one without them.

## 3. Seed the cloud databases (from your laptop)

The seed scripts are deterministic (fixed random seed), so the cloud data matches the
local data exactly. Environment variables override `.env`:

```bash
POSTGRES_SEED_URL='postgresql://neondb_owner:...@ep-xxxx.../bookstore?sslmode=require' \
MONGO_SEED_URL='mongodb+srv://talktable_seed:...@cluster0.abcde.mongodb.net/?retryWrites=true&w=majority' \
.venv/bin/python -m data.seed.seed_all
```

**Check** that it ends with `Seeded: sqlite, postgres, mongo.`. Then confirm the read-only URLs work
and really are read-only:

```bash
export POSTGRES_URL='postgresql://talktable_ro:...'   MONGO_URL='mongodb+srv://talktable_ro:...'
.venv/bin/python -m talktable.cli --status            # all three online
.venv/bin/python -c "
from talktable.config import get_settings; import psycopg
with psycopg.connect(get_settings().postgres_url.get_secret_value()) as c:
    c.execute('CREATE TABLE should_fail (i int)')"  # must fail: read-only transaction
```

## 4. An LLM API key

Pick one:

- **Groq** (recommended: free, and answers in a few seconds). Create a key at
  [console.groq.com/keys](https://console.groq.com/keys). Then set `LLM_PROVIDER=groq` and
  `LLM_MODEL=llama-3.3-70b-versatile`.
- **Hugging Face Inference Providers.** Create a fine-grained token at
  [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with *Make calls
  to Inference Providers* checked. Then set `LLM_PROVIDER=hf` and
  `LLM_MODEL=Qwen/Qwen2.5-72B-Instruct`. Free accounts get only a small monthly credit.

**Check** it locally before deploying. Comment out `LLM_QUERY_MODEL` in `.env` first: it
names a local Ollama model that Groq and Hugging Face don't have.

```bash
LLM_PROVIDER=groq LLM_MODEL=llama-3.3-70b-versatile GROQ_API_KEY=gsk_... \
.venv/bin/python -m talktable.cli "How many orders were placed each month in 2025?"
```

## 5. Create the Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space). Name it `talktable`,
   license MIT, SDK **Gradio**, template **Blank**, hardware **CPU basic (free)**, Public.
2. Open the Space's **Settings**, then **Variables and secrets**, and add:

   | Name | Type | Value |
   |---|---|---|
   | `POSTGRES_URL` | Secret | the Neon **read-only** URL |
   | `MONGO_URL` | Secret | the Atlas **read-only** URL |
   | `GROQ_API_KEY` or `HF_TOKEN` | Secret | your API key or token |
   | `LLM_PROVIDER` | Variable | `groq` (or `hf`) |
   | `LLM_MODEL` | Variable | `llama-3.3-70b-versatile` (or `Qwen/Qwen2.5-72B-Instruct`) |
   | `MONGO_DB` | Variable | `bookstore` |

   Never add the seed URLs to the Space, and leave `LLM_QUERY_MODEL` unset.
   Variables are publicly visible and secrets are not, so anything with a password
   goes in a secret.

## 6. Upload the code

Hugging Face rejects binary files (`data/bookstore.db`, `docs/screenshot.png`) in a plain
`git push`, so upload with the `hf` CLI, which handles them. Upload from a fresh clone, so
only committed files go up: never `.env`, never `.venv`.

```bash
.venv/bin/hf auth login                         # paste a token with write access
rm -rf /tmp/talktable-space && git clone --depth 1 "file://$PWD" /tmp/talktable-space
.venv/bin/hf upload <your-hf-username>/talktable /tmp/talktable-space . \
    --repo-type space --exclude ".git/*" --commit-message "Deploy TalkTable"
```

The Space builds by itself. **Check** the **Logs** tab. The build installs
`requirements.txt`, the CPU-only PyTorch included, which takes a few minutes. Startup
should then log
`online databases: sqlite, postgres, mongo` and `Running on local URL: http://0.0.0.0:7860`.

To redeploy after changes: commit, then repeat the three commands.

## 7. Try it

- **The UI:** `https://huggingface.co/spaces/<your-hf-username>/talktable`. Click a sidebar
  example; the database badges, query, and results table should appear within seconds.
- **The MCP endpoint:** `https://<your-hf-username>-talktable.hf.space/gradio_api/mcp`
  (streamable HTTP). The page footer's "Use via API or MCP" link shows the same five tools.
  Clients that support remote MCP servers can use the URL directly. For Claude Desktop,
  [`mcp-remote`](https://www.npmjs.com/package/mcp-remote) bridges it to stdio (needs
  Node.js on the machine running Claude Desktop):

  ```json
  {
    "mcpServers": {
      "talktable-space": {
        "command": "npx",
        "args": ["-y", "mcp-remote", "https://<your-hf-username>-talktable.hf.space/gradio_api/mcp"]
      }
    }
  }
  ```

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| A database shows *offline* in the sidebar | Wrong secret, Atlas network access not `0.0.0.0/0`, or Neon project suspended. Click **Check connections** after fixing it. |
| *The groq API rejected the credentials* | `GROQ_API_KEY` missing or mistyped (secrets apply after a restart: **Settings**, then **Restart**). |
| *Model '…' was not found* | `LLM_MODEL` doesn't exist on that provider, or `LLM_QUERY_MODEL` is set to an Ollama model. |
| *rate limit was reached* | The Groq free tier allows a limited number of requests per minute; wait a minute. |
| Build fails installing `torch` | Keep the `--extra-index-url` line at the top of `requirements.txt` (CPU-only wheels). |
| First question after a while is slow | Free Spaces sleep when idle, and Neon suspends idle databases. The first request wakes both up. |
