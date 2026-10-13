# Using TalkTable as an MCP server

`mcp_server.py` runs TalkTable as a **stdio** MCP server. The client (MCP Inspector,
Claude Desktop, Cursor, ...) starts it as a subprocess and talks to it over
stdin/stdout. Logs go to stderr.

Before you start:

```bash
docker/dev-db.sh up          # local Postgres + MongoDB (SQLite needs nothing)
ollama serve                 # if Ollama isn't already running as a service
cp .env.example .env         # once; then adjust if needed
```

## The tools

| Tool | Arguments | Returns |
|---|---|---|
| `list_databases` | – | Each database: name, kind, online status, description, tables/collections |
| `get_schema` | `database` | Tables or collections with columns/fields, types, descriptions, keys, row counts |
| `ask` | `question` | `AskResult`: answer, databases used, query, rows, retries, latency (`error` set if no answer) |
| `run_sql` | `database`, `query` | Rows from your own `SELECT` on `sqlite` or `postgres`, after the guardrails |
| `run_mongo` | `collection`, `pipeline` (JSON array) | Documents from your own aggregation on `mongo`, after the guardrails |

All five are marked read-only (`readOnlyHint`). Hand-written queries go through the
same guardrails as generated ones, and the databases themselves are opened read-only.
A blocked query comes back as a tool error that explains why, e.g.
`Blocked by the read-only guardrails: Only SELECT queries are allowed, not DROP.`

`ask` reports progress while it works ("Choosing a database", "Writing the query", ...,
plus a heartbeat every 5 seconds) to clients that ask for progress. With the local
models a question takes about 20-70 s (see `.env.example` for the measured models).
An MCP client can also skip the local LLM entirely: `get_schema`, then write the query
itself with `run_sql` / `run_mongo`. With Claude as the client that is quicker, and it
can combine data from several databases.

The web app (`app.py`) serves the same five tools over HTTP at
`http://127.0.0.1:7860/gradio_api/mcp`, and the deployed Space at
`https://<your-hf-username>-talktable.hf.space/gradio_api/mcp` (see [deploy.md](deploy.md)).
This page is about the stdio server.

## 1. MCP Inspector

[MCP Inspector](https://github.com/modelcontextprotocol/inspector) is the official
tool for trying out an MCP server. These steps were tested with Inspector 2.10, which
needs Node.js 22.19 or newer (24 is installed here).

**Mind the 60-second limit.** The Inspector stops waiting for a tool after 60 seconds.
Every tool except `ask` answers instantly. With the default local models (qwen3:14b +
qwen2.5-coder:7b) `ask` takes about a minute, so the commands below add
`-e LLM_MODEL=qwen2.5-coder:7b`, which answers in about 20-25 s. Put `-e` *after*
`mcp_server.py`. The Inspector saves `-e` values in `~/.mcp-inspector/secrets.json`,
so never pass real secrets this way. TalkTable reads its own `.env` and doesn't need them.

**Web UI.** From the project root:

```bash
npx @modelcontextprotocol/inspector .venv/bin/python mcp_server.py -e LLM_MODEL=qwen2.5-coder:7b
```

It prints `MCP Inspector Web is up and running at: http://127.0.0.1:6274?MCP_INSPECTOR_API_TOKEN=...`.
Open that link in a browser on Windows (WSL forwards localhost), connect, list the tools
(all five appear), and call them. For example:

1. `run_sql` with `database` = `sqlite` and
   `query` = `SELECT title, price FROM books ORDER BY price DESC LIMIT 5`.
2. `run_sql` with `DROP TABLE books`, which is rejected by the guardrails.
3. `ask` with `How many orders were placed each month in 2025?`.

**Command line.** The same checks without a browser:

```bash
npx @modelcontextprotocol/inspector --cli .venv/bin/python mcp_server.py --method tools/list

npx @modelcontextprotocol/inspector --cli .venv/bin/python mcp_server.py \
  --method tools/call --tool-name run_sql \
  --tool-arg database=sqlite \
  --tool-arg 'query=SELECT title, price FROM books ORDER BY price DESC LIMIT 5'

npx @modelcontextprotocol/inspector --cli .venv/bin/python mcp_server.py \
  --method tools/call --tool-name run_mongo \
  --tool-arg collection=reviews \
  --tool-arg 'pipeline=[{"$match": {"rating": 1}}, {"$group": {"_id": "$book_id", "n": {"$sum": 1}}}, {"$sort": {"n": -1}}, {"$limit": 5}]'

npx @modelcontextprotocol/inspector --cli .venv/bin/python mcp_server.py -e LLM_MODEL=qwen2.5-coder:7b \
  --method tools/call --tool-name ask \
  --tool-arg 'question=How many orders were placed each month in 2025?'
```

## 2. Claude Desktop (Windows, server in WSL)

Claude Desktop runs on Windows, and TalkTable runs inside WSL, so the config starts the
server through `wsl.exe`.

1. In Claude Desktop open **Settings → Developer → Edit Config**. This opens
   `claude_desktop_config.json`. For the Microsoft Store version of the app it is at
   `%LOCALAPPDATA%\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Roaming\Claude\claude_desktop_config.json`.
   Otherwise it is at `%APPDATA%\Claude\claude_desktop_config.json`.
2. Add an `mcpServers` section with the `talktable` entry. Keep whatever is already in
   the file, such as `"preferences"`, and separate the sections with a comma:

   ```json
   {
     "mcpServers": {
       "talktable": {
         "command": "wsl.exe",
         "args": [
           "-d", "Ubuntu-20.04",
           "-e", "/home/<you>/talktable/.venv/bin/python",
           "/home/<you>/talktable/mcp_server.py"
         ]
       }
     }
   }
   ```

3. Quit Claude Desktop completely (also from the system tray) and start it again.
   In a new chat, TalkTable's five tools show up in the tools menu.
4. Ask something like *"Using TalkTable, which 5 books have the most 1-star reviews,
   and what are their titles?"* Claude can call `ask`, or combine `run_mongo` (the
   reviews) with `run_sql` (the titles) by itself.

On macOS or Linux, without WSL, use the Python path directly:

```json
{
  "mcpServers": {
    "talktable": {
      "command": "/path/to/multi_database/.venv/bin/python",
      "args": ["/path/to/multi_database/mcp_server.py"]
    }
  }
}
```

The server reads `.env` from the project folder, so the working directory doesn't matter.

### Troubleshooting

- **Server log.** Claude Desktop writes the server's stderr to `mcp-server-talktable.log`
  in the `logs` folder next to the config file. MCP Inspector shows it in the bottom panel.
- **A database shows as offline.** Start it (`docker/dev-db.sh up`). The next
  `list_databases` call reconnects automatically, so there's no need to restart.
- **`ask` is slow or times out.** Use a faster local model
  (`LLM_MODEL=qwen2.5-coder:7b` with `LLM_QUERY_MODEL` empty, ~20 s) or a hosted one
  (`LLM_PROVIDER=groq`, a few seconds). Or let Claude write the queries itself with
  `get_schema` + `run_sql` / `run_mongo`.
- **Nothing happens at all.** Run the server by hand. It should sit there waiting
  for input (Ctrl+C to stop):
  `wsl.exe -d Ubuntu-20.04 -e /home/<you>/talktable/.venv/bin/python /home/<you>/talktable/mcp_server.py`
