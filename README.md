
# Chat with Audio RAG

## Description

This project enables users to interact with audio content through a chat interface using Retrieval-Augmented Generation (RAG). It likely transcribes audio files, stores the transcriptions in a vector database (Qdrant), and allows users to ask questions or chat about the audio content, leveraging Large Language Models (LLMs) accessed via LlamaIndex.

## Features (Example - Please update based on actual functionality)

*   Transcribe audio files using OpenAI's Whisper.
*   Index transcriptions into a Qdrant vector store.
*   Perform RAG-based chat/Q&A over audio content.
*   Interactive user interface built with Streamlit.

## Prerequisites

*   [Docker](https://docs.docker.com/get-docker/) installed and running.
*   Python 3.8+

## Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <your-repository-url>
    cd chat-with-audio-rag
    ```

2.  **Set up a Python virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install git+https://github.com/openai/whisper.git
    pip install transformers datasets torchaudio streamlit llama-index-vector-stores-qdrant llama-index-llms-sambanovasystems sseclient-py langchain
    ```
    *Alternatively, if you have a `requirements.txt` file:*
    ```bash
    pip install -r requirements.txt
    ```

4.  **Start the Qdrant Vector Store:**
    Run the following Docker command to start a Qdrant instance. This command mounts a local directory (`qdrant_storage`) to persist the vector data.
    ```bash
    docker run -p 6333:6333 -p 6334:6334 \
        -v $(pwd)/qdrant_storage:/qdrant/storage:z \
        qdrant/qdrant
    ```
    *(Note: Ensure the `qdrant_storage` directory exists in your project root or adjust the path accordingly. The `:z` flag is often used for SELinux systems; it might not be necessary depending on your OS.)*

## Configuration

*(Add details here about any necessary configuration, such as:*
*   *API keys for LLMs (e.g., SambaNova)*
*   *Qdrant connection details (if not using defaults)*
*   *Paths to audio files or directories)*
*   *Environment variables)*

*Example:*
Create a `.env` file in the project root with the following variables:
```
SAMBANOVA_API_KEY=your_api_key_here
QDRANT_URL=http://localhost:6333
```

## Usage

*(Add instructions on how to run the application. This usually involves running a Python script or a Streamlit command.)*

*Example:*
1.  **Load data (if necessary):**
    *(Explain how to process and index the audio files)*
    ```bash
    python process_audio.py --audio-dir /path/to/your/audio
    ```
2.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
3.  Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

## Dependencies

*   [OpenAI Whisper](https://github.com/openai/whisper): Audio transcription.
*   [Transformers](https://huggingface.co/docs/transformers/index): ML models and tools.
*   [Datasets](https://huggingface.co/docs/datasets/index): Handling datasets.
*   [TorchAudio](https://pytorch.org/audio/stable/index.html): Audio processing library.
*   [Streamlit](https://streamlit.io/): Web application framework.
*   [LlamaIndex](https://www.llamaindex.ai/): Data framework for LLM applications (including Qdrant integration, SambaNova LLM support).
*   [Qdrant](https://qdrant.tech/): Vector database.
*   [Langchain](https://www.langchain.com/): Framework for developing applications powered by language models.
*   [sseclient-py](https://pypi.org/project/sseclient-py/): Server-Sent Events client.

---

*(Optional: Add sections for Contributing, License, Acknowledgements)*

