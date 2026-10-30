"""The same instructions and message format for training and comparison."""

system_instruction = (
    "Answer only from approved context. Start with the direct answer, then "
    "give the next action. Cite every policy claim with source IDs in square "
    "brackets. If the context is insufficient, reply exactly: "
    "The approved material is insufficient. Escalate for manual review."
)


def build_messages(question, context):
    return [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": f"Approved context:\n{context}\n\nQuestion: {question}"},
    ]
