from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt
# Replace with your actual token (inside quotes)
hf_token="-----"
# Create the HuggingFaceEndpoint first to represent the model
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",  # Choose your model
    task="text-generation",
    huggingfacehub_api_token=hf_token,
      provider="hf-inference"   # API token for HuggingFace
)

# Pass the LLM to ChatHuggingFace
model = ChatHuggingFace(
    llm=llm  # This is the missing required field
)
st.header('Resarch Tool')
paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )
style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 
length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )
template=load_prompt('template.json')
if st.button('Summarize'):
    chain=template|model
    result=chain.invoke({'paper_input':paper_input,
                  'style_input':style_input,
                  'length_input':length_input})
    st.write(result.content)
