import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Technical RAG Assistant", page_icon="🤖", layout="centered")

st.title("🤖 Technical Documentation RAG Assistant")
st.write("Upload your technical notes, document them in the system, and chat with your AI agent.")

st.sidebar.header("Document Management")
uploaded_file = st.sidebar.file_uploader("Upload a document", type=["txt", "md", "pdf"])

if uploaded_file is not None:
    if st.sidebar.button("Ingest File"):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        try:
            response = requests.post(f"{BACKEND_URL}/ingest", files=files)
            if response.status_code == 200:
                st.sidebar.success(f"Successfully ingested {uploaded_file.name}!")
            else:
                st.sidebar.error("Failed to ingest document.")
        except Exception as e:
            st.sidebar.error(f"Connection error: {e}")


if st.sidebar.button("List Indexed Documents"):
    try:
        res = requests.get(f"{BACKEND_URL}/documents")
        if res.status_code == 200:
            docs = res.json().get("indexed_documents", [])
            st.sidebar.write("**Indexed Sources:**")
            for doc in docs:
                st.sidebar.text(f"- {doc}")
    except Exception:
        st.sidebar.error("Could not reach backend.")


session_id = st.text_input("Session ID (for chat memory)", value="default_session")

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Ask a technical question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                payload = {"question": prompt, "session_id": session_id}
                res = requests.post(f"{BACKEND_URL}/query", json=payload)
                
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("answer")
                    sources = data.get("sources", [])
                    
                    st.markdown(answer)
                    
                    if sources:
                        st.caption(f"**Sources:** {', '.join(sources)}")
                        
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error("Error communicating with the backend.")
            except Exception as e:
                st.error(f"Failed to connect to FastAPI server: {e}")