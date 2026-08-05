# lang-writer

Small LangGraph demo running against the local OpenAI-compatible server on the DGX box
(`http://10.150.0.30:1234/v1`).

The graph is a draft/review/revise loop:

```
START -> plan -> write -> review --(approved or max revisions)--> END
                   ^________________|  (otherwise: revise)
```

The model id is discovered from `/v1/models` at startup (the lineup on the box changes
often — don't hardcode it).

## Run

```sh
uv run main.py "your topic here"
```

## Config (env vars)

| Var              | Default                       |
|------------------|-------------------------------|
| `MODEL_BASE_URL` | `http://10.150.0.30:1234/v1` |
| `MODEL_ID`       | first model from `/v1/models` |
| `MAX_REVISIONS`  | `2`                           |
