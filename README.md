# Agentic Expense Tracker

A comprehensive AI-powered financial assistant that helps users track their purchases and expenses through multiple interfaces including a web chat interface and WhatsApp integration. The bot can process text descriptions, receipt images, and voice messages to automatically extract and store purchase data.


https://github.com/user-attachments/assets/238527fb-1692-41fd-8084-8e5b60a52362



## 🚀 Features

- **Multi-Interface Support**: Web interface (Chainlit), WhatsApp integration
- **Multiple Input Methods**: 
  - Text descriptions of purchases
  - Receipt image analysis
  - Voice message transcription
- **Multi-Agent Architecture**: 
  - **Main Agent**: Orchestrates conversations and saves receipt data.
  - **Database Analyst Agent**: Translates natural language into SQL queries for advanced spending analytics.
- **Two-Tier Memory System**: 
  - **Short-term Memory** (Supabase): Contextual awareness of the current conversation.
  - **Long-term Memory** (Qdrant): Semantic vector embeddings for cross-session recall.
- **Hybrid Search & Advanced Database**: PostgreSQL (via Supabase) with `pgvector` for semantic item matching and advanced SQL sandboxing to prevent LLM exploits.
- **Voice Synthesis**: Text-to-speech responses using ElevenLabs.
- **Observability**: Built-in Langfuse integration for end-to-end trace tracking.
- **Flexible Architecture**: Easily switch between AI providers (Gemini, Groq, etc.).

## 🏗️ Architecture

The project follows a port-adapter architecture with dependency injection:

1. **Agents** (`src/agents/`): Multi-agent system (Main Agent & Database Analyst Agent).
2. **Memory Manager** (`src/adapters/memory/`): Orchestrator for two-tier memory (Supabase for short-term, Qdrant for long-term).
3. **Ports** (`src/ports/`): Abstract interfaces defining contracts for services (LLM, STT, TTS, Vision, Database).
4. **Adapters** (`src/adapters/`): Concrete implementations of ports (Gemini, Groq, ElevenLabs, Supabase, Qdrant, SQLite).
5. **Domain Models** (`src/domain/`): Business logic independent of external services.
6. **Interfaces** (`src/interfaces/`): User-facing components (Chainlit web, WhatsApp webhook).7. **Dependency Injection Container** (`src/config/containers.py`): Manages provider instances and configuration.

## 📋 Prerequisites

- Python 3.12+
- API Keys for:
  - Google Gemini (for AI processing, Embeddings, and Speech-to-Text)
  - Groq (optional, for LLM/Vision)
  - ElevenLabs (for text-to-speech)
  - WhatsApp Business API (for WhatsApp integration)
- Database credentials for:
  - Supabase (PostgreSQL)
  - Qdrant (Vector Database)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/TharangaSG/agentic-expense-tracker.git
   cd agentic-expense-tracker
   ```

2. **Install dependencies using uv**:
   ```bash
   uv sync
   ```

3. **Activate the virtual environment**:
   ```bash
   source .venv/bin/activate  # On Linux/Mac
   # or
   .venv\Scripts\activate     # On Windows
   ```

4. **Set up environment variables**:
   Copy `.env.example` to `.env` and fill in your API keys and database endpoints.
   ```bash
   cp .env.example .env
   ```

5. **Optional: Enable Langfuse tracing**
   See [Langfuse Setup Guide](docs/setup_langfuse.md) to capture traces, token usage, and model costs.


## 🚀 Usage

### Web Interface (Chainlit)

Start the web interface:
```bash
chainlit run src/interfaces/chainlit/app.py
```
Available at `http://localhost:8000`

### WhatsApp Integration

1. Set up WhatsApp API (see [WhatsApp Setup Guide](docs/setup_whatsapp.md))
2. Start the server:
   ```bash
   python run_whatsapp.py
   # or
   uvicorn src.interfaces.whatsapp.whatsapp_app:app --host 0.0.0.0 --port 8001
   ```


## 📊 Example Usage

### Text Input
```
"I bought 3 apples for $2 each and 2 bananas for $1.50 total"
```

### Voice & Image Input
- Send a voice note: "I spent $25 on groceries today..."
- Upload a receipt photo for automatic extraction.

### Spending Queries (Database Analyst Agent)
```
"How much money have I spent on Biscuits?"
"What were my top 5 expenses this month?"
"Did I spend more on vegetables or meat?"
```

## 🗄️ Database Schema

The application primarily uses PostgreSQL (via Supabase) with the `pgvector` extension for semantic search:

```sql
CREATE TABLE items (
    id SERIAL PRIMARY KEY,
    receipt_id INTEGER NOT NULL,
    item_name TEXT NOT NULL,
    quantity REAL NOT NULL,
    unit_price REAL NOT NULL,
    total_price REAL NOT NULL,
    purchase_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    item_name_embedding vector(768)
);
```

## 🐳 Docker Deployment

Build and run using Docker:

```bash
# Build the image
docker build -t financial-assistant .

# Run the WhatsApp webhook server
docker run -p 8001:8001 --env-file .env financial-assistant
```

## 🔧 Configuration

### Model Settings

Configure AI providers and memory settings in `.env`:

- **LLM Provider**: `LLM_PROVIDER` (options: `gemini`, `groq`)
- **Vision Provider**: `VISION_PROVIDER` (options: `groq`)
- **Speech-to-Text Provider**: `STT_PROVIDER` (options: `gemini`)
- **Text-to-Speech Provider**: `TTS_PROVIDER` (options: `elevenlabs`)
- **Database Provider**: `DATABASE_PROVIDER` (options: `postgres`, `sqlite`)

### Memory Configuration

- `MEMORY_ENABLED=True`
- Supabase details (`SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`) for Short-Term Memory
- Qdrant details (`QDRANT_URL`, `QDRANT_API_KEY`) for Long-Term Semantic Memory

### Langfuse Observability

Set up Langfuse variables (`LANGFUSE_ENABLED`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`) to monitor traces, nested multi-agent interactions, and costs.

## 🆘 Support

For setup, see:
- [WhatsApp Setup Guide](docs/setup_whatsapp.md)
- [Memory System Architecture](docs/Memory%20Syatem.md)

## 🔮 Future Enhancements

- Budget tracking and threshold alerts
- Allow users to cancel or delete an entered transaction within a 5-minute window
- Advanced proactive analytics and reporting summaries
- Enhanced receipt categorization
