# NKK-GPT

## 1. Overview

NKK-GPT is a powerful intelligent agent system integrating multiple advanced capabilities:

### 1.1 Core Features
- **Code generation & review**
- **Intelligent file operations**
- **Multi-model collaborative processing**
- **Sandbox environment execution**
- **Git version control integration**

### 1.2 Technical Highlights
- **Tool management** via MCP (Model Control Protocol)
- **Asynchronous parallel processing architecture**
- **Multi-agent collaboration system**
- **Secure sandbox execution environment**

---

## 2. Installation

Clone the project from GitHub using:

```bash
git clone https://github.com/2025NKUCS-agent/NKK-GPT.git
```

### Requirements:
- **Python Version**: Python 3.9+ (recommended)
- **Dependencies**: Install required packages via:

```bash
pip install -r requirements.txt
pip install -e .
```

---

## 3. Running the Project

### Configuration:
Set parameters (API keys, model names, etc.) in the `config` directory.

### Launch:
Start the system with:

```bash
nkkagent
```

---

## 4. Project Structure

```
nkkagent/
├── agent/         # Core agent logic  
├── config/        # Configuration management  
├── llm/           # Language model integration  
├── mcp/           # Model Control Protocol  
├── sandbox/       # Sandbox environment  
└── tools/         # Tool collection
```

---

## 5. Contribution Guidelines

To contribute:
1. Fork the repository
2. Create a new development branch
3. Submit a Pull Request with your changes
