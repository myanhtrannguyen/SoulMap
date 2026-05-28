import json


def format_chart_context(
    user_chart: dict | None,
    tuvi_chart: dict | None,
    houses_chart: list[dict],
) -> str:
    return json.dumps(
        {
            "user_chart": user_chart,
            "tuvi_chart": tuvi_chart,
            "houses_chart": houses_chart,
        },
        ensure_ascii=False,
        indent=2,
    )


def build_initial_prompt(
    houses_chart: list[dict],
    retrieved_docs: list[dict],
    user_chart: dict | None = None,
    tuvi_chart: dict | None = None,
) -> str:
    context = "\n\n".join(
        f"[{i+1}] {doc['chunk_text']}"
        for i, doc in enumerate(retrieved_docs)
    )

    chart_context = format_chart_context(user_chart, tuvi_chart, houses_chart)

    return f"""
Bạn là hệ thống phân tích Tử Vi theo dữ liệu đã cung cấp.

Nhiệm vụ:
- Dựa trên lá số và các luận giải được truy xuất.
- Tóm tắt những điểm nổi bật ban đầu của người dùng.
- Không khẳng định tuyệt đối.
- Trình bày theo hướng tham khảo, dễ hiểu.
- Không tự bịa thêm ngoài context.

LÁ SỐ:
{chart_context}

CONTEXT TRUY XUẤT:
{context}

Hãy trả lời gồm:
1. Tổng quan nổi bật
2. Tính cách/khuynh hướng
3. Công việc/học vấn
4. Tài chính
5. Tình cảm
6. Lưu ý cần cân bằng
""".strip()

def build_followup_prompt(
    user_query: str,
    houses_chart: list[dict],
    initial_summary: str,
    retrieved_docs: list[dict],
    user_chart: dict | None = None,
    tuvi_chart: dict | None = None,
) -> str:

    context = "\n\n".join(
        f"[{i+1}] {doc['chunk_text']}"
        for i, doc in enumerate(retrieved_docs)
    )

    chart_context = format_chart_context(user_chart, tuvi_chart, houses_chart)

    return f"""
Bạn là hệ thống hỏi đáp Tử Vi dựa trên RAG.

Quy tắc:
- Chỉ dùng lá số, đặc điểm ban đầu và context truy xuất.
- Nếu context chưa đủ, nói rõ là chưa đủ dữ liệu.
- Không phán đoán tuyệt đối.
- Trả lời đúng trọng tâm câu hỏi.

LÁ SỐ:
{chart_context}

ĐẶC ĐIỂM NỔI BẬT BAN ĐẦU:
{initial_summary}

CÂU HỎI NGƯỜI DÙNG:
{user_query}

CONTEXT TRUY XUẤT:
{context}

Hãy trả lời bằng tiếng Việt, rõ ràng, có phân tích theo từng ý.
""".strip()
