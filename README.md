# Agentic Expense Tracker

A comprehensive AI-powered receipt processing and financial tracking system that extracts, structures, and analyzes receipt data using multiple LLM providers and database storage solutions.

## 🚀 Overview

This project provides an intelligent receipt processing pipeline that can:
- Extract text from receipt images using computer vision
- Structure unorganized receipt data using AI language models
- Store receipt information in SQLite database
- Query spending patterns and provide financial insights
- Support multiple AI providers (Gemini, Groq/Llama) for different tasks

## 🏗️ Architecture & Components

### Core Modules

1. **`read_image.py`** - Image Processing & OCR
   - Extracts text from receipt images (local files or URLs)
   - Uses Groq API with Llama-4-Scout model for vision tasks
   - Supports base64 encoding for image processing

2. **`data_inserting_flow.py`** - Receipt Processing Workflow
   - Orchestrates the complete receipt processing pipeline
   - Uses Google Gemini for text extraction and data structuring
   - Implements function calling with structured outputs
   - Stores data in SQLite database

3. **`data_fetching_flow.py`** - Conversational AI Interface
   - Provides natural language interface for querying spending data
   - Implements tool calling for database queries
   - Uses Gemini for conversational responses

4. **`database_query_tool.py`** - Database Query Functions
   - Implements spending analysis functions
   - Supports fuzzy matching for item names
   - Provides formatted responses for AI consumption

## 📋 Prerequisites

- Python 3.12+
- API Keys for:
  - Google Gemini (`GEMINI_API_KEY`)
  - Groq (`GROQ_API_KEY`)
- PostgreSQL database (optional, for advanced features)

## 🔧 Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   # or if using uv:
   uv sync
   ```

3. Set up environment variables in `.env`:
   ```bash
   GEMINI_API_KEY=your_gemini_api_key
   GROQ_API_KEY=your_groq_api_key
 