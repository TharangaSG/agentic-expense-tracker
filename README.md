# Financial Assistant Bot

A comprehensive AI-powered financial assistant that helps users track their purchases and expenses through multiple interfaces including a web chat interface and WhatsApp integration. The bot can process text descriptions, receipt images, and voice messages to automatically extract and store purchase data.


https://github.com/user-attachments/assets/238527fb-1692-41fd-8084-8e5b60a52362



## 🚀 Features

- **Multi-Interface Support**: Web interface (Chainlit) and WhatsApp integration
- **Multiple Input Methods**: 
  - Text descriptions of purchases
  - Receipt image analysis
  - Voice message transcription

- **AI-Powered Processing**: Uses Google Gemini for intelligent data extraction
- **Database Storage**: SQLite database for persistent purchase tracking
- **Spending Analytics**: Query spending patterns and totals
- **Voice Synthesis**: Text-to-speech responses using ElevenLabs
- **Flexible Architecture**: Easily switch between different AI providers (Gemini, Groq, etc.) via configuration

## 🏗️ Architecture

The project follows a port-adapter architecture with dependency injection, consisting of:

1. **Ports** (`src/ports/`): Abstract interfaces defining contracts for services (LLM, STT, TTS, Vision, Database)
2. **Adapters** (`src/adapters/`): Concrete implementations of ports for specific services (Gemini, Groq, ElevenLabs, SQLite)
3. **Domain Models** (`src/domain/`): Business logic independent of external services
4. **Interfaces** (`src/interfaces/`): User-facing components (Chainlit web, WhatsApp webhook)
5. **Dependency Injection Container** (`src/config/containers.py`): Manages provider instances and configuration

The project consists of two main applications:

1. **Chainlit Web Interface** (`src/interfaces/chainlit/app.py`): Interactive web chat interface
2. **WhatsApp Bot** (`src/interfaces/whatsapp/`): FastAPI-based webhook server for WhatsApp integration

## 📋 Prerequisites

- Python 3.12+
- API Keys for:
  - Google Gemini (for AI processing and speech-to-text)
  - ElevenLabs (for text-to-speech)
  - WhatsApp Business API (for WhatsApp integration)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/TharangaSG/agentic-expense-tracker.git
   cd financial-assistance
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
   Copy `.env.example` to `.env` and fill in your API keys:
   ```bash
   cp .env.example .env
   ```


## 🚀 Usage

### Web Interface (Chainlit)

Start the web interface:
```bash
chainlit run src/interfaces/chainlit/app.py
```

The web interface will be available at `http://localhost:8000`

**Features:**
- Type purchase descriptions
- Upload receipt images
- Record voice messages 
- Real-time audio processing with voice responses

### WhatsApp Integration

1. **Set up WhatsApp Business API** (see [WhatsApp Setup Guide](docs/setup_whatsapp.md))

2. **Start the WhatsApp webhook server**:
   ```bash
   python run_whatsapp.py
   ```
   
   Or using uvicorn directly:
   ```bash
   uvicorn src.interfaces.whatsapp.whatsapp_app:app --host 0.0.0.0 --port 8001
   ```

3. **Configure webhook URL**: `https://your-domain.com:8001/whatsapp_response`

**Supported WhatsApp message types:**
- Text messages with purchase descriptions
- Receipt images
- Voice messages

### Standalone Scripts

**Data Insertion Flow**:
```bash
python src/data_inserting_flow.py
```

**Data Fetching/Query Flow**:
```bash
python src/data_fetching_flow.py
```

## 📊 Example Usage

### Text Input
```
"I bought 3 apples for $2 each and 2 bananas for $1.50 total"
```

### Voice Input
Record yourself saying: "I spent $25 on groceries today, bought milk, bread, and eggs"

### Image Input
Upload a photo of a receipt, and the AI will automatically extract item details.

### Spending Queries
```
"How much money have I spent on Biscuits?"
```

## 🗄️ Database Schema

The application uses SQLite with the following schema:

```sql
CREATE TABLE items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    receipt_id INTEGER,
    item_name TEXT,
    quantity REAL,
    unit_price REAL,
    total_price REAL
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

The application uses a dependency injection container to manage different service providers. You can configure which providers to use by setting environment variables in your `.env` file:

- **LLM Provider**: `LLM_PROVIDER` (options: `gemini`, `groq`)
- **Vision Provider**: `VISION_PROVIDER` (options: `groq`)
- **Speech-to-Text Provider**: `STT_PROVIDER` (options: `gemini`)
- **Text-to-Speech Provider**: `TTS_PROVIDER` (options: `elevenlabs`)
- **Database Provider**: Configured automatically (currently SQLite only)

Example configuration in `.env`:
```
LLM_PROVIDER=groq
VISION_PROVIDER=groq
STT_PROVIDER=gemini
TTS_PROVIDER=elevenlabs
```

### Available Providers
- **LLM**: Gemini (`gemini-2.5-flash`) or Groq (`llama-3.1-8b-instant`, `llama-3.1-70b-versatile`, etc.)
- **Vision**: Groq Vision models for image analysis
- **Speech-to-Text**: Gemini for audio transcription
- **Text-to-Speech**: ElevenLabs for voice synthesis
- **Database**: SQLite for persistent storage

### Benefits of Container Architecture
- **Easy Provider Switching**: Change AI providers without modifying code
- **Environment Flexibility**: Different configurations for dev, staging, production
- **Testability**: Easy to mock providers during testing
- **Maintainability**: Clean separation of concerns between business logic and service implementations
- **Extensibility**: Simple to add new providers by implementing the appropriate port interface


## 🆘 Support

For WhatsApp integration setup, see the detailed guide: [WhatsApp Setup Guide](docs/setup_whatsapp.md)

For issues and questions, please open an issue in the repository.

## 🔮 Future Enhancements

- Add Chat memory
- Retrieve stored data through WhatsApp
- Advanced analytics and reporting
- Receipt categorization
- Budget tracking and alerts
