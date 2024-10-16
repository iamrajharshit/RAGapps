# Doc Wise: Engage with PDFs
- Retrival-Augmented Generation (RAG) application using Gemini-pro and Langchain to enable conversational interaction with PDFs. 
- Utilized GoogleGenerativeAIEmbeddings for high-dimensional text vectors and Chroma as a vector store for efficient document chunking and semantic search.

## LangChain Architechture
<img src="https://github.com/iamrajharshit/RAGapps/blob/main/assets/img/LangChain%20for%20RAG.jpg">

The core components and processes involved in this LangChain application are:

**1. Document Loading:**
* **Input:** PDF documents
* **Process:** The system loads the PDF documents into memory.

**2. Chunking:**
* **Process:** The PDF documents are divided into smaller chunks or segments to make them more manageable for processing and to facilitate efficient retrieval.

**3. Embedding:**
* **Process:** Each chunk of text is converted into a numerical representation called an embedding. Embeddings capture the semantic meaning of the text, allowing the system to understand and compare different pieces of text.

**4. Vector Store:**
* **Process:** The embeddings of all the chunks are stored in a vector store, which is essentially a database optimized for storing and retrieving high-dimensional vectors.

**5. Question Embedding:**
* **Input:** User query
* **Process:** The user's query is also converted into an embedding, similar to the document chunks. This allows the system to compare the query's meaning to the embeddings in the vector store.

**6. Semantic Search:**
* **Process:** The system performs a similarity search between the query embedding and the embeddings stored in the vector store. This identifies the chunks of text that are most relevant to the query based on their semantic similarity.

**7. Ranked Results:**
* **Output:** The most relevant chunks of text are returned to the user, ranked in order of their similarity to the query.

## Demo:

<img src="https://github.com/iamrajharshit/RAGapps/blob/main/assets/Demo.gif" width="1080" height="480" />
