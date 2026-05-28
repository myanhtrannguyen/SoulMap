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

## Retrieval modes

Tham so `--retrieval-mode` chon pipeline retrieve. Hien co 5 mode:

| Mode | Du lieu dung | Cach retrieve | Khi nen dung |
| --- | --- | --- | --- |
| `fixed_similarity` | `rag_documents_fixed_chunks.jsonl` | Chi semantic similarity tren fixed-size chunks. | Baseline don gian tren van ban dai, khong loc theo la so. |
| `fixed_structured_similarity` | Fixed chunks + `rag_documents_tu_vi_boi_toan.jsonl` | Semantic similarity tren fixed chunks va structured docs da loc theo data structure cua la so. | Muon ket hop tai lieu tong hop va rules co cau truc. |
| `fixed_structured_keyword` | Fixed chunks + `rag_documents_tu_vi_boi_toan.jsonl` | Ket hop semantic text va semantic keyword/data fields. Day la mode mac dinh. | Mode day du nhat, can ca noi dung tong quat va keyword theo cung/sao/chu de. |
| `structured_similarity` | `rag_documents_tu_vi_boi_toan.jsonl` | Chi semantic similarity tren `chunk_text` cua structured docs da loc theo data structure cua la so. | Muon retrieve rieng tu bo data co cau truc. |
| `structured_keyword` | `rag_documents_tu_vi_boi_toan.jsonl` | Chi semantic similarity tren cac field keyword/data structure: `topic`, `palace_id`, `palace_name`, `star_id`, `required_stars`, `context_type`, `condition`. | Muon retrieve theo keyword va data structure, khong dua vao fixed chunks. |

Vi du chay tung mode:

```powershell
python src/main.py "Duong tinh duyen cua toi nhu the nao?" --retrieval-mode fixed_similarity
python src/main.py "Duong tinh duyen cua toi nhu the nao?" --retrieval-mode fixed_structured_similarity
python src/main.py "Duong tinh duyen cua toi nhu the nao?" --retrieval-mode fixed_structured_keyword
python src/main.py "Duong tinh duyen cua toi nhu the nao?" --retrieval-mode structured_similarity
python src/main.py "Duong tinh duyen cua toi nhu the nao?" --retrieval-mode structured_keyword
```

Khi host retriever server, co the chon mode luc start server:

```powershell
python -m src.rag.retriever_server --retrieval-mode structured_keyword
```

Co the dat bien moi truong de khoi phai truyen flag moi lan:

```powershell
$env:RETRIEVER_URL="http://127.0.0.1:8765"
python src/main.py "Duong tinh duyen cua toi nhu the nao?"
```
