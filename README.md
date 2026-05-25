# SoulMap
AI-powered birth chart interpretation assistant

## Host retriever

Chay retriever server mot lan de giu SentenceTransformer va embeddings trong memory:

```powershell
python -m src.rag.retriever_server
```

Sau do chay main va tro toi server:

```powershell
python src/main.py "Duong tinh duyen cua toi nhu the nao?" --retriever-url http://127.0.0.1:8765
```

Co the dat bien moi truong de khoi phai truyen flag moi lan:

```powershell
$env:RETRIEVER_URL="http://127.0.0.1:8765"
python src/main.py "Duong tinh duyen cua toi nhu the nao?"
```
