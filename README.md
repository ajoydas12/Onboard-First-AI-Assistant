
-----

# Onboard-First AI Assistant

This repository contains the source code for a minimal, end-to-end AI assistant for Occams Advisory. The assistant is designed to answer questions based on the company's website and to onboard prospective clients by collecting their contact information in a secure, conversational manner.

-----

## Architecture Diagram 🏗️

The system follows a simple client-server architecture with an offline scraping process to build the knowledge base.

```ascii
+-----------------+      HTTPS/JSON       +---------------------------------+
|                 | <-------------------> |                                 |
|   Frontend      |      /chat API        |    Backend Server (Python/Flask)|
|  (HTML/JS/CSS)  |                       |                                 |
|                 |                       |  [State & PII Handling Logic]   |
+-----------------+                       +---------------------------------+
                                                     |           ^
                                                     |           | PII NEVER SENT
                                                     V           |
                                              [Q&A Logic]        | API Call
                                                     |           |(Query + Context)
                                                     |           |
+----------------------+                             V           |
|                      |      Local Read      +---------------------------------+
|  Knowledge Base      | <------------------+ |                                 |
| (FAISS Index + JSON) |                      |  3rd Party LLM API (e.g., Gemini) |
|                      |                      |                                 |
+----------------------+                      +---------------------------------+
      ^
      | Offline Process
      |
+----------------------+
|                      |
|  Scraper (scrape.py) |
|                      |
+----------------------+
```

-----

## Key Design Choices & Trade-offs

### Unified Chat Interface

A single, stateful chat interface was chosen to handle both general questions and the onboarding process. When a user interacts with the chatbot, the backend server tracks their state (e.g., `idle`, `awaiting_name`, `complete`). This creates a fluid, conversational experience where the assistant can intelligently pivot from answering a question to starting the sign-up flow.

  * **Trade-off:** This approach adds complexity to the backend logic compared to using simple, separate web forms for Q\&A and sign-ups. However, the significant improvement in user experience was prioritized, as it makes the interaction feel more natural and engaging, which is crucial for a client-facing tool.

### Local Retrieval-Augmented Generation (RAG)

To ensure all answers about Occams Advisory are accurate and grounded in facts, a Retrieval-Augmented Generation (RAG) architecture was implemented. The website content is scraped, converted into numerical vector embeddings, and stored in a local `FAISS` vector database. When a user asks a question, the system first performs a semantic search to find the most relevant text chunks from the website. These chunks are then injected into the prompt as context for the LLM.

  * **Trade-off:** This design requires an initial, offline scraping and indexing step. The alternative—directly asking an LLM questions about the company—is simpler but highly unreliable and prone to **hallucination** (making up facts). By grounding every answer with retrieved context, we gain massive improvements in accuracy and trustworthiness, directly addressing a core project constraint.

-----

## Threat Model & PII Mitigation 🛡️

  * **PII Flow:** Personally Identifiable Information (Name, Email, Phone) is sent from the user's browser to our backend server.
  * **Asset at Risk:** The user's PII.
  * **Threat:** The primary threat is the accidental transmission of this sensitive PII to a third-party service, specifically the LLM API.
  * **Mitigation Strategy:** The backend employs a **strict separation of concerns**. The application's state machine distinguishes between a Q\&A request and an onboarding data submission. If the user is in an onboarding state (e.g., `awaiting_email`), their input is processed by local validation functions and stored in the server-side session. **This logic path never makes an external API call.** Only general queries, which are stripped of any potential PII, are sent to the LLM along with the publicly available website context. This ensures PII never leaves our controlled server environment.

-----

## Scraping Approach

The knowledge base was created by transforming the unstructured data from the `occamsadvisory.com` website into a structured, searchable format.

1.  **Fetch & Parse:** The `scrape.py` script uses the `requests` library to fetch the raw HTML of the website. `BeautifulSoup` then parses this HTML, systematically extracting clean text from meaningful tags like `<p>`, `<h1>`, `<h2>`, and `<li>` while ignoring irrelevant code and navigation elements.
2.  **Chunk & Store:** The extracted text is segmented into smaller, semantically coherent chunks (e.g., by paragraph). These raw text chunks are saved in `knowledge_base.json`, which serves as the ground truth for our system.
3.  **Vectorize & Index:** Each text chunk is converted into a high-dimensional vector embedding using the `sentence-transformers` model. These vectors, which represent the semantic meaning of the text, are stored in a `FAISS` index file. This index allows for extremely fast and efficient similarity searches, enabling the bot to instantly find the most relevant context for any given user question.

-----

## Failure Modes & Graceful Degradation

The system is designed to handle potential failures gracefully without crashing or providing a poor user experience.

  * **Primary Failure Mode: LLM API Unavailability**
    The most likely failure is the inability to connect to the external LLM API due to network issues, an invalid API key, or the service being down. The backend code wraps all API calls in a `try...except` block. If an exception occurs, the system **degrades gracefully**. Instead of showing an error, it falls back to providing the single most relevant text chunk retrieved from the local `FAISS` search. The user receives a helpful, contextually relevant snippet directly from the website with a message explaining the temporary issue, ensuring the app remains useful even when partially offline.

  * **Secondary Failure Mode: Irrelevant Questions**
    If a user asks a question for which no relevant information exists on the website (e.g., "What's the weather like in Paris?"), the vector search will still retrieve the "closest" matching chunks, which will be irrelevant. This is mitigated through **prompt engineering**. The prompt sent to the LLM explicitly instructs it to answer *only* based on the provided context and to state that it doesn't have the information if the context is not useful. This prevents the LLM from hallucinating and ensures it honestly reports the limits of its knowledge base.