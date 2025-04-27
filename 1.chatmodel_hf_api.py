from langchain_huggingface.chat_models import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint

# Replace with your actual token (inside quotes)
hf_token = "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"

# Create the HuggingFaceEndpoint first to represent the model
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",  # Choose your model
    task="text-generation",
    huggingfacehub_api_token=hf_token  # API token for HuggingFace
)

# Pass the LLM to ChatHuggingFace
model = ChatHuggingFace(
    llm=llm  # This is the missing required field
)

# Now invoke the model
result = model.invoke("Who is the PM of India?")
print(result.content)