# LLM-Routing

## Banana API routing

Banana API is OpenAI-compatible, so the API script calls:

```text
POST https://api.banana2556.com/v1/chat/completions
Authorization: Bearer <BANANA_API_KEY>
```

Create a token in the Banana console, then set these environment variables in
PowerShell:

```powershell
$env:BANANA_API_KEY="sk-your-token-here"
$env:BANANA_BASE_URL="https://api.banana2556.com/v1"
$env:BANANA_MODEL="google/gemma-3-4b-it"
```

If the model name in your token's model restriction list is different, use that
exact string for `BANANA_MODEL`.

Preview the first prompt without making an API request:

```powershell
python routing_gemma_api_judge.py --dry-run
```

Run a small smoke test:

```powershell
python routing_gemma_api_judge.py --limit 3 --out submission_gemma_api_smoke.csv
```

Run the full test set:

```powershell
python routing_gemma_api_judge.py --out submission_gemma_api.csv --request-log api_responses.jsonl --resume
```
