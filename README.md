# DeBotte AI - Employee Skills Finder

An intelligent RAG-based chatbot that helps managers find the best-suited employees for projects by analyzing One-Pager documents.

## 🚀 Features

- **Smart Document Processing**: Extracts employee information from PPTX files using advanced NLP
- **Advanced RAG System**: Uses FAISS vector search with semantic embeddings for accurate candidate matching
- **Interactive Chat Interface**: Beautiful Streamlit-based chat interface with chat history
- **Context-Aware Search**: Maintains conversation context and provides relevant follow-up suggestions
- **Multi-Session Support**: Save and manage multiple chat sessions

## 🏗️ Architecture

The system consists of three main components:

1. **Document Processing** (`docling-one-pagers/`): Extracts structured data from PPTX files
2. **RAG Engine** (`rag/`): Processes data, creates embeddings, and builds FAISS index
3. **Frontend** (`frontend/`): Streamlit-based chat interface

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key
- PPTX files with employee information

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd RAG-Chatbot-for-One-Pagers
   ```

2. **Install dependencies**:
   ```bash
   # Install frontend dependencies
   pip install -r frontend/requirements.txt
   
   # Install additional dependencies for RAG
   pip install numpy ujson tqdm python-dotenv openai faiss-cpu
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## 🚀 Quick Start

### Option 1: Run Complete Pipeline (Recommended)

Use the automated pipeline runner:

```bash
python run_pipeline.py
```

This will:
1. Extract data from PPTX files
2. Build the RAG system
3. Start the frontend

### Option 2: Manual Step-by-Step

#### Step 1: Process PPTX Files
```bash
cd docling-one-pagers
python employee_rag_extractor.py
```

#### Step 2: Build RAG System
```bash
cd rag
python main.py auto
```

#### Step 3: Start Frontend
```bash
cd frontend
streamlit run app.py
```

## 📁 File Structure

```
RAG-Chatbot-for-One-Pagers/
├── docling-one-pagers/          # Document processing
│   ├── employee_rag_extractor.py
│   ├── input_dir/               # PPTX files
│   └── json_output/             # Extracted JSON data
├── rag/                         # RAG engine
│   ├── main.py                  # Main pipeline runner
│   ├── convert_json_to_chunks.py
│   ├── embed_only.py            # Embedding generation
│   └── faiss_service.py         # FAISS search service
├── frontend/                    # Streamlit interface
│   ├── app.py                   # Main application
│   ├── requirements.txt
│   └── faiss_store/             # FAISS index storage
└── run_pipeline.py              # Automated pipeline runner
```

## 🔍 Usage Examples

### Chat Interface Queries

- **Skills-based search**: "Find employees with AWS certification"
- **Experience search**: "Who has experience with SAP S/4HANA?"
- **Role-based search**: "Find project managers with cloud experience"
- **Certification search**: "Show me Azure certified administrators"
- **Client experience**: "Who has worked with financial services clients?"

### Follow-up Questions

- "Tell me more about [employee name]"
- "What other skills does [employee name] have?"
- "Compare [employee1] and [employee2]"

## 🧠 How It Works

1. **Document Processing**: PPTX files are processed to extract structured employee data
2. **Chunking**: Employee data is split into meaningful chunks (summary, skills, experience, etc.)
3. **Embedding**: Text chunks are converted to vector embeddings using OpenAI's text-embedding-3-large
4. **Indexing**: FAISS index is built for fast similarity search
5. **Search**: User queries are embedded and matched against the index
6. **Ranking**: Results are ranked using cosine similarity with boosting for exact matches
7. **Response**: Relevant candidates are presented with confidence scores

## ⚙️ Configuration

### RAG Parameters

- **Embedding Model**: `text-embedding-3-large` (default)
- **Chunk Size**: Optimized for employee information
- **Search Parameters**: Top-k retrieval with MMR diversity
- **Boosting**: Special handling for certifications and exact phrase matches

### Frontend Settings

- **Chat History**: Persistent across sessions
- **Result Display**: Top 5 candidates with confidence scores
- **UI Theme**: Dark mode optimized for professional use

## 🔧 Troubleshooting

### Common Issues

1. **FAISS Index Not Found**
   - Run: `cd rag && python main.py auto`
   - Check that `frontend/faiss_store/` contains index files

2. **OpenAI API Error**
   - Verify your API key in `.env` file
   - Check API quota and billing status

3. **Import Errors**
   - Ensure all dependencies are installed
   - Check Python path and virtual environment

4. **Memory Issues**
   - Reduce batch size in `embed_only.py`
   - Use smaller embedding model if needed

### Debug Mode

Enable verbose logging:
```bash
cd rag
python main.py --verbose
```

## 📊 Performance

- **Indexing**: ~1000 employees in ~2-3 minutes
- **Search**: Sub-second response time for most queries
- **Accuracy**: High precision with semantic understanding
- **Scalability**: Supports thousands of employee profiles

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with Streamlit, LangChain, and FAISS
- Powered by OpenAI's embedding models
- Developed by Innov8 team

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the code comments
3. Open an issue on GitHub

---

**Happy Employee Hunting! 🎯**