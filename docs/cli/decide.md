# pmetal decide

Answer typed questions about a state with a decision model.

A decision model reads a state (text or any JSON value) and a schema of typed questions, and
returns a probability for every allowed option of every question from one forward pass, with no
text generation. PMetal runs the Clef releases, `Cloudflare/clef` (Qwen3.8-27B backbone) and
`Cloudflare/clef-flash` (Qwen3.5-9B-class backbone). The request and response are the
`/v1/systemone` bodies, which `pmetal serve` also answers when it serves a decision model.

## Usage

```bash
pmetal decide --model <MODEL> --request <request.json | -> [OPTIONS]
```

| Option | Default | Meaning |
|---|---|---|
| `--model` | | Decision model ID or path |
| `--request` | | Request body as a JSON file, or `-` for stdin |
| `--max-length` | 16384 | Longest prompt in tokens; the state is truncated to fit |
| `--compact` | off | Print the response on one line |

## Request

```json
{
  "model": "clef-flash",
  "state": "Our checkout started returning errors and orders are blocked.",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle the message?",
      "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"}
    },
    "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
    "outage": {"type": "noul", "instructions": "Is a service down?"}
  }
}
```

Each question has a `type`:

- `noul`: a proposition; the answer is the probability that it is true. `criteria` may override
  the descriptions of `true` and `false`.
- `choice`: one of the named options in `criteria` (option id to description).
- `score`: one of the ordered levels in `criteria` (a list, indexed from 0); the answer is the
  expected level.

`instructions` is optional; the question id stands in for it.

## Response

```json
{
  "model": "clef-flash",
  "answers": {
    "department": {"type": "choice", "choice": "technical", "confidence": 0.9607,
                   "probabilities": {"billing": 0.0393, "technical": 0.9607}},
    "urgency": {"type": "score", "score": 1.7873, "confidence": 0.8584,
                "legend": {"0": "Can wait", "1": "This week", "2": "Today"},
                "probabilities": {"0": 0.071, "1": 0.0706, "2": 0.8584}},
    "outage": {"type": "noul", "noul": 0.8362}
  },
  "usage": {"input_tokens": 300, "output_tokens": 0}
}
```

## Examples

```bash
pmetal decide --model Cloudflare/clef-flash --request request.json

# Many requests: serve the model once
pmetal serve --model Cloudflare/clef-flash --port 8080
curl http://localhost:8080/v1/systemone \
  -H "Content-Type: application/json" -d @request.json
```

:::note
A record's `images` are base64 strings, data URIs or, in `pmetal decide` only, file paths; its
`videos` are lists of frames, or `{"frames": [...], "fps": N}`. Video files are not decoded. The
backbone must be unquantized: the head reads option vectors from its LM head.
:::

## See Also

- [pmetal serve](/cli/serve/) — Serve a decision model on `/v1/systemone`
- [pmetal mcp](/cli/mcp/) — The `decide` MCP tool, which loads the model for one call, or with
  `port` asks a decision model `start_serve` is already serving
