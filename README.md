# LangChain RAG Q&A Agent

A retrieval-augmented Q&A agent built with LangChain 1.x. It answers questions from a small
set of petroleum-sector documents, using semantic search rather than keyword matching, and
remembers earlier turns in the conversation.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shyampandey263/langchain-rag-qa/blob/main/langchain-rag-qa.ipynb)

## How it works
1. Documents are loaded, split into chunks, and converted into embeddings (nomic-embed-text).
2. Chunks are stored in a Chroma vector database.
3. A question triggers a retriever to find the most relevant chunks by meaning.
4. The chunks are passed to an LLM (llama3.1) along with the question, and it answers only
   from that context.
5. The whole pipeline is wrapped as a tool inside an agent, with memory across turns, so a
   follow-up question like "Which ministry is it under?" correctly resolves "it" to the
   subject of the previous answer.

## The six LangChain components, and where they appear
| Component | Where |
|---|---|
| Models | `ChatOllama` (llama3.1) for answers, `OllamaEmbeddings` (nomic-embed-text) for search |
| Prompts | The `ChatPromptTemplate` that restricts answers to the given context |
| Chains | The LCEL `rag_chain`: retriever → format → prompt → model → parser |
| Memory | `InMemorySaver` + `thread_id`, so follow-up questions work |
| Indexes | `TextLoader` → `RecursiveCharacterTextSplitter` → `Chroma` vector store → retriever |
| Agents | `create_agent`, wrapping the RAG chain as a single tool |

## How to run
1. Click the "Open In Colab" badge above.
2. Runtime, then Change runtime type, then T4 GPU.
3. Run the cells top to bottom. It installs Ollama, downloads llama3.1 and nomic-embed-text,
   builds the vector store from the sample documents, and runs the agent.

## Known limitation
The RAG chain itself refuses to guess ("I don't know") when a question isn't covered by the
documents. But the agent wrapped around it isn't told to enforce that same rule, so it can
fall back on the model's own general knowledge after the tool reports no match (for example,
answering "What is the capital of France?" correctly even though that's not in the documents).
Tightening the agent's system prompt to refuse in that case is a natural next step.

## Sample documents
The three `.txt` files are small placeholder documents about Indian petroleum pricing, PPAC,
and diesel consumption, used to demonstrate the pipeline. Swapping in real documents only
requires changing the file list in the loading cell.

## Tech
Python, LangChain 1.x, LangGraph, Ollama (llama3.1, nomic-embed-text), ChromaDB, Google Colab
