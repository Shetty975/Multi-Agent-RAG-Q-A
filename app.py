
# app.py

import streamlit as st
from tools import agent_router

# Page config
st.set_page_config(page_title="Multi-Agent RAG Q&A Assistant")

# Title
st.title("📚 Multi-Agent RAG Q&A")

# Layout with search bar and file uploader
col1, col2 = st.columns([5, 1])

with col1:
    query = st.text_input("Ask me anything...")

with col2:
    uploaded_files = st.file_uploader("📂", type=["pdf", "docx", "txt"], accept_multiple_files=True, label_visibility="collapsed")

# Show uploaded files (optional display for debug or future ingestion)
if uploaded_files:
    st.markdown("### 📥 Uploaded Documents:")
    for file in uploaded_files:
        st.markdown(f"- {file.name}")

# Process query if provided
if query:
    with st.spinner("Thinking..."):
        try:
            response = agent_router(query)

            st.success("✅ Answer generated!")
            st.markdown(f"🔁 Agent Path Chosen: **{response['route']}**")

            if 'context' in response:
                st.markdown("**📄 Top Retrieved Chunks:**")
                for i, chunk in enumerate(response['context']):
                    st.markdown(f"**Chunk {i+1}:** {chunk}")

            st.markdown(f"**💬 Final Answer:** {response['result']}")
        
        except Exception as e:
            st.error(f"⚠️ Error occurred: {e}")

# Optional: Show debug logs
if st.checkbox('Show Debug Log', False):
    with open('agent.log', 'r') as log_file:
        st.text_area("Debug Log", log_file.read(), height=300)
