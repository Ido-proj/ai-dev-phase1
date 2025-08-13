# ai-dev-phase1

AI Dev Tools &amp; Ecosystem Deep Dive Goal:
Be fully comfortable with the frameworks, libraries, and infrastructure that power AI agents.

--------------------------------------------------------------------------------------------------

**Week 1–2: Hugging Face & Transformers**

Why:
Almost every modern AI application (including agents) relies on pretrained models — Hugging Face is the hub.

What to Learn:
  1. Hugging Face Basics
    - Install and set up account (pip install transformers datasets huggingface_hub)
    - Explore Hugging Face Model Hub — find a model for NLP, CV, and embeddings.
    - Transformers Fundamentals
  
  2. Tokenization (AutoTokenizer)
    - Model loading (AutoModel, AutoModelForSequenceClassification)
    - Inference pipelines (pipeline() for text generation, classification, embeddings)
  
  3. Practical Mini-Projects
    - Sentiment analysis API (Flask + Hugging Face model)
    - Embedding generator + FAISS search
  
  📚 Resources:
      -Hugging Face course (free)
      -Docs: transformers library
      
--------------------------------------------------------------------------------------------------

Week 3: LangChain Fundamentals

Why:
This is the glue for chaining LLM reasoning, memory, and tools — core for agents.

What to Learn:

  1. LangChain Fundamentals
    - LLM wrappers (OpenAI, HuggingFaceHub)
    - Prompt templates
    - Chains (Sequential, MapReduce)
    - Tools & Agents (Tool calling)
    - Memory types (ConversationBuffer, VectorStore memory)

  2. Practical Mini-Projects
    -Build a Q&A bot with memory
    -Connect an agent to call a weather API

📚 Resources:
  - LangChain docs
  - LangChain YouTube crash course by Sam Witteveen

--------------------------------------------------------------------------------------------------
Week 4: LlamaIndex (Optional but Powerful)
Why:
If LangChain is a Swiss Army knife, LlamaIndex is the librarian — it organizes and retrieves data for LLMs.

What to Learn:
  1. Data Connectors
    - Local file ingestion (PDF, CSV)
    - Web scraping connector
  2. Indexes
    - Vector indexes
    - Keyword indexes
  3. RAG (Retrieval-Augmented Generation)
    - Load documents → Embed → Query with LLM
     
Mini-Project
“Ask my Docs” chatbot from your own PDFs

📚 Resources:
  - LlamaIndex docs
  - “Building RAG Apps” by Jerry Liu (YouTube)

--------------------------------------------------------------------------------------------------

Week 5: Vector Databases

Why:
Agents need memory and retrieval — vector DBs store embeddings for that.

What to Learn:
  1. FAISS (local)
    - Install: pip install faiss-cpu
    - Create embeddings with OpenAI / Hugging Face
    - Store and search vectors
     
  2. Cloud Vector DBs
    - Pinecone (pip install pinecone-client)
    - Weaviate or Milvus basics

Mini-Project:
Build a semantic search API for research papers

📚 Resources:
  - FAISS docs
  - Pinecone learn
--------------------------------------------------------------------------------------------------

Week 6: Putting It All Together

Why: 
Build a full pipeline to glue the tools together.

Integration Project: Research Assistant Agent

Stack:
  - Hugging Face for embeddings
  - FAISS for memory
  - LangChain agent for reasoning
  - Tool to search Wikipedia

Flow:
  1. User asks question
  2. Agent retrieves relevant chunks from FAISS
  3. Agent decides if it needs external search
  4. Agent answers with cited sources
--------------------------------------------------------------------------------------------------

Extra Tools to Explore Later
Weights & Biases — experiment tracking for AI projects

Gradio / Streamlit — rapid UI for AI apps

Docker — containerize your agents

OpenTelemetry — observability for multi-step agents
