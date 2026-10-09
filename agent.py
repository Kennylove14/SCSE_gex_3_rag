### COMPLETE THE CODE ###

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rag_tool import KnowledgeBaseSearchTool

# Local model path (relative to the RAG project folder)
MODEL_PATH = "D:/software/2026 SCSE/SCSE_gex_3/models/qwen3-0.6b"

# Load model and tokenizer (CPU mode, local files only)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype="auto",
    local_files_only=True,
    device_map="cpu"
)
model.eval()

kb_tool = KnowledgeBaseSearchTool()

SYSTEM_INSTRUCTION = """You are a university IT support assistant.
Answer ONLY using information found in the provided knowledge base context.
If the answer is not present in the context, clearly state that the information is not available and you do not know the answer.
Never invent, guess or make up phone numbers, links or facts.
Keep all answers short and concise."""


def generate_answer(user_question: str) -> str:
    # Step 1: Retrieve relevant policy context
    context = kb_tool.run(user_question)

    # Step 2: Build full prompt
    full_prompt = f"""{SYSTEM_INSTRUCTION}

Knowledge base context:
{context}

User question: {user_question}

Assistant:"""

    # Step 3: Generate response (greedy decoding, no hallucination)
    inputs = tokenizer(full_prompt, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )
    
    full_response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    assistant_reply = full_response.split("Assistant:")[-1].strip()
    return assistant_reply


# Test the target question
if __name__ == "__main__":
    test_query = "What exact phone number should I call for the University IT Service Desk?"
    result = generate_answer(test_query)
    print("Model response:", result)
