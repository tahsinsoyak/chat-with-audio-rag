# chat-with-audio-rag

 pip install git+https://github.com/openai/whisper.git

pip install transformers datasets torchaudio

pip install streamlit  llama-index-vector-stores-qdrant llama-index-llms-sambanovasystems sseclient-py

pip install langchain

docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant