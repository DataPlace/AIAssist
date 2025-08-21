# pages/1_AI_Assistant.py
import streamlit as st
from modules import parser, embeddings, vectorstore, rag, genai_client, persistence, emailer
import numpy as np

st.set_page_config(page_title="AI Assistant", layout="centered")
st.title("AI Assistant — Ask & Create Anything")

# Initialize embedder
if 'embedder' not in st.session_state:
    st.session_state['embedder'] = embeddings.Embedder()

if 'faiss' not in st.session_state:
    st.session_state['faiss'] = None

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Upload & Index
with st.expander("Upload & Index Documents"):
    uploaded = st.file_uploader("Upload PDF / DOCX / TXT (multiple)", accept_multiple_files=True)
    if uploaded:
        st.write("Files to parse:")
        for f in uploaded:
            st.write(f.name)
        if st.button("Parse & Index Uploaded Files"):
            texts = []
            metadatas = []
            for f in uploaded:
                raw = f.getvalue()
                try:
                    text = parser.parse_file(f.name, raw)
                except Exception as e:
                    st.error(f"Failed to parse {f.name}: {e}")
                    continue
                chunks = parser.chunk_text(text)
                for c in chunks:
                    meta = {"source": f.name, "text": c["text"]}
                    metadatas.append(meta)
                    texts.append(c["text"])
            if texts:
                embedder = st.session_state['embedder']
                embs = embedder.embed(texts)
                vs = vectorstore.FaissIndex(dim=embs.shape[1])
                vs.add(embs, metadatas)
                st.session_state['faiss'] = vs
                st.success(f"Indexed {len(texts)} chunks")

st.markdown("---")

# Chat UI
query = st.text_area("Ask anything or give creation instructions", height=120)
model_name = st.text_input("GROQ model name (optional)", value="")
if st.button("Send") and query.strip():
    emb = st.session_state['embedder'].embed([query])
    retrieved = []
    if st.session_state.get('faiss'):
        retrieved = st.session_state['faiss'].search(emb, k=5)[0]
    prompt = rag.assemble_prompt(query, retrieved)
    response = genai_client.generate_with_groq(prompt, max_tokens=512, temperature=0.7, model=model_name or None)
    st.session_state['chat_history'].append({'query': query, 'response': response, 'retrieved': retrieved})
    persistence.save_conversation(query, response, retrieved)
    st.experimental_rerun()

# Display conversation (most recent first)
st.markdown("### Conversation")
for msg in reversed(st.session_state.get('chat_history', [])):
    st.markdown("**User:**")
    st.write(msg['query'])
    st.markdown("**Assistant:**")
    st.write(msg['response'])
    if msg.get('retrieved'):
        with st.expander("Show retrieved sources"):
            for r in msg['retrieved']:
                meta = r['metadata']
                st.write(f"Source: {meta.get('source')}, score: {r.get('score'):.3f}")
                st.write(meta.get('text')[:800] + ("..." if len(meta.get('text',''))>800 else ""))

st.markdown("---")

# Email section
st.markdown("### Send an assistant response by email")
if st.session_state.get('chat_history'):
    last = st.session_state['chat_history'][-1]['response']
else:
    last = ""
recipient = st.text_input("Recipient email")
subject = st.text_input("Subject", value="From Smart Creation Chatbot")
body = st.text_area("Email body", value=st.session_state.get('email_body', last), height=200)
if st.button("Send Email"):
    if not recipient:
        st.error("Provide recipient email")
    else:
        res = emailer.send_email(subject, body, recipient)
        if res.get('ok'):
            st.success("Email sent")
        else:
            st.error("Failed to send: " + str(res.get('error')))
