import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class RemoteRetriever:
    def __init__(self, base_url: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        request = Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Retriever server HTTP {exc.code}: {body}") from exc
        except URLError as exc:
            raise RuntimeError(f"Khong ket noi duoc retriever server: {exc.reason}") from exc

    def search(
        self,
        query: str,
        chart_index: dict | None = None,
        top_k: int = 8,
        alpha: float = 0.75,
        mode: str | None = None,
    ) -> list[dict]:
        response = self._post(
            "/search",
            {
                "query": query,
                "chart_index": chart_index,
                "top_k": top_k,
                "alpha": alpha,
                "mode": mode,
            },
        )
        return response["results"]
