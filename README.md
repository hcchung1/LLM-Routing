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

List models available to your token:

```powershell
python routing_gemma_api_judge.py --list-models
```

Run a small smoke test:

```powershell
python routing_gemma_api_judge.py --limit 3 --out submission_gemma_api_smoke.csv
```

Run the full test set:

```powershell
python routing_gemma_api_judge.py --out submission_gemma_api.csv --request-log api_responses.jsonl --resume
```

For cheaper/smaller API models, add retrieved examples from the training set:

```powershell
python routing_gemma_api_judge.py --model claude-haiku-4.5-as --retrieval-shots 8 --shots-per-label 0 --out submission_haiku_retrieval_api.csv --request-log api_responses_haiku_retrieval.jsonl --resume
```

For ARC-AGI rows that only contain an opaque task ID, use the training-family
prior instead of spending an API call:

```powershell
python routing_gemma_api_judge.py --model gpt-5.5 --retrieval-shots 8 --shots-per-label 0 --arc-agi-prior-override --out submission_gpt55_retrieval_api.csv --request-log api_responses_gpt55_retrieval.jsonl --resume
```

## Google AI Studio Gemma routing

Google AI Studio uses the Gemini API format, not the OpenAI-compatible Banana
format. Set a Google AI Studio key:

```powershell
$env:GEMINI_API_KEY="your-google-ai-studio-key"
$env:GEMINI_MODEL="gemma-4-31b-it"
```

Run a small smoke test:

```powershell
python routing_gemma_api_judge.py --provider google --limit 3 --out submission_google_gemma_smoke.csv
```

Run the retrieval version:

```powershell
python routing_gemma_api_judge.py --provider google --model gemma-4-31b-it --retrieval-shots 8 --shots-per-label 0 --arc-agi-prior-override --out submission_google_gemma_api.csv --request-log api_responses_google_gemma.jsonl --resume
```

The script omits Google `thinkingConfig` by default because `gemma-4-31b-it`
does not support thinking budget configuration.
