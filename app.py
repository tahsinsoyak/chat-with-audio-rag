import os
import uuid
import gc
import tempfile
import streamlit as st
import time
from rag_utils import EmbedData, QdrantVDB_QB, Retriever, RAG, Transcribe
import logging

# Suppress HuggingFace cache warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}
    st.session_state.messages = []
    st.session_state.transcripts = []
    st.session_state.query_engine = None
    st.session_state.last_file_info = None  # Track last processed file

session_id = st.session_state.id
collection_name = "chat_with_audios"
batch_size = 32
vector_dim = 1024  # Matches intfloat/multilingual-e5-large

def reset_chat():
    """Reset chat history and clear memory."""
    st.session_state.messages = []
    st.session_state.transcripts = []
    st.session_state.query_engine = None
    st.session_state.last_file_info = None
    st.session_state.file_cache.clear()
    gc.collect()

with st.sidebar:
    st.header("Ses dosyanızı ekleyin!")
    uploaded_file = st.file_uploader("Ses dosyanızı seçin", type=["mp3", "wav", "m4a"])

    if uploaded_file:
        # Validate file size
        if uploaded_file.size > 200 * 1024 * 1024:
            st.error("Dosya boyutu 200MB sınırını aşıyor.")
        else:
            # Check if the file is new or different
            current_file_info = (uploaded_file.name, uploaded_file.size)
            if st.session_state.last_file_info != current_file_info:
                with st.spinner("Ses dosyası işleniyor..."):
                    # Create a temporary file that persists
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}")
                    try:
                        # Write uploaded file content to temp file
                        temp_file.write(uploaded_file.getvalue())
                        temp_file.close()  # Close to ensure file is accessible
                        file_path = temp_file.name

                        file_key = f"{session_id}-{uploaded_file.name}"
                        st.write("AssemblyAI ile transkribe ediliyor...")

                        # Initialize transcription
                        api_key = os.getenv("ASSEMBLYAI_API_KEY")
                        if not api_key:
                            st.error("AssemblyAI API anahtarı bulunamadı. Lütfen ASSEMBLYAI_API_KEY ortam değişkenini ayarlayın.")
                        else:
                            try:
                                transcriber = Transcribe(api_key=api_key)
                                transcripts = transcriber.transcribe_audio(file_path)
                                st.session_state.transcripts = transcripts

                                # Prepare documents
                                documents = [f"{t['speaker']}: {t['text']}" for t in transcripts]
                                if not documents:
                                    st.error("Transkripsiyon boş. Ses dosyasında içerik olmayabilir.")
                                else:
                                    # Embed data
                                    with st.spinner("Metin gömülüyor..."):
                                        embeddata = EmbedData(embed_model_name="intfloat/multilingual-e5-large", batch_size=batch_size)
                                        embeddata.embed(documents)

                                    # Set up Qdrant
                                    with st.spinner("Vektör veritabanına kaydediliyor..."):
                                        qdrant_vdb = QdrantVDB_QB(collection_name=collection_name, batch_size=batch_size, vector_dim=vector_dim)
                                        qdrant_vdb.define_client()
                                        qdrant_vdb.reset_collection()  # Reset to avoid duplicates
                                        qdrant_vdb.ingest_data(embeddata)

                                    # Set up RAG
                                    with st.spinner("Sorgu motoru başlatılıyor..."):
                                        retriever = Retriever(vector_db=qdrant_vdb, embeddata=embeddata)
                                        query_engine = RAG(retriever=retriever, llm_model="Llama-4-Scout-17B-16E-Instruct")
                                        st.session_state.query_engine = query_engine
                                        st.session_state.last_file_info = current_file_info

                                    st.success("Sohbete hazır!")
                                    st.audio(uploaded_file)

                                    # Display transcript
                                    st.subheader("Transkript")
                                    with st.expander("Tam transkripti göster", expanded=True):
                                        for t in transcripts:
                                            st.text(f"**{t['speaker']}**: {t['text']}")

                            except Exception as e:
                                st.error(f"İşlem başarısız: {str(e)}")
                                logger.error(f"Processing failed: {e}")
                            finally:
                                # Clean up temporary file
                                try:
                                    os.unlink(file_path)
                                except Exception as e:
                                    logger.warning(f"Failed to delete temp file {file_path}: {e}")
                    except Exception as e:
                        st.error(f"Dosya yazma başarısız: {str(e)}")
                        logger.error(f"File write failed: {e}")
                        if os.path.exists(temp_file.name):
                            os.unlink(temp_file.name)
            else:
                st.info("Bu ses dosyası zaten işlendi. Soru sorabilirsiniz.")
                if st.session_state.transcripts:
                    st.subheader("Transkript")
                    with st.expander("Tam transkripti göster", expanded=True):
                        for t in st.session_state.transcripts:
                            st.text(f"**{t['speaker']}**: {t['text']}")
                if uploaded_file:
                    st.audio(uploaded_file)

col1, col2 = st.columns([6, 1])
with col1:
    st.markdown(
        """
        # Ses Üzerinden RAG: LangChain ve Qdrant Destekli
        Bu uygulama, ses dosyalarınızla RAG modeli kullanarak sohbet etmenizi sağlar.
        """, unsafe_allow_html=True
    )

with col2:
    st.button("Temizle ↺", on_click=reset_chat)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user queries
if prompt := st.chat_input("Ses kaydı hakkında sorularınızı sorun..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            if not st.session_state.query_engine:
                raise ValueError("No audio file has been processed yet. Please upload an audio file.")
            full_response = st.session_state.query_engine.query(prompt)
            displayed_response = ""
            for char in full_response:
                displayed_response += char
                message_placeholder.markdown(displayed_response)
                time.sleep(0.05)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            st.error(f"Sorgu başarısız: {str(e)}")
            logger.error(f"Query failed: {e}")