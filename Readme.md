# 📚 Multi-Agent RAG Q&A

---

## 🌟 Project Overview

**Multi-Agent RAG Q&A** is an intelligent knowledge assistant designed to revolutionize how users interact with textual information. It integrates **Retrieval-Augmented Generation (RAG)** and an **agentic decision-making workflow** to deliver smart, context-aware answers to natural language questions.

This assistant doesn’t just retrieve and respond — it **reasons, routes, and learns**.

---

## 🔍 Problem Statement

Modern users seek instant, reliable answers across complex and unstructured documents. Traditional search engines or static chatbots often **fail to understand context** or dynamically route tasks.

The need is clear: build a system that can:
- Retrieve accurate information from relevant sources
- Interpret user intent intelligently
- Delegate tasks to appropriate computational tools — autonomously

---

## 🧠 Introduction

This project explores the fusion of:
- **Large Language Models (LLMs)**
- **Vector-based document retrieval**
- **Agent-based reasoning**

Together, they enable a seamless Q&A experience. Through a **multi-agent orchestration**, the assistant can:
- Pull facts from indexed knowledge bases
- Define complex terms using dictionary APIs
- Perform mathematical computations when asked

The goal is to **emulate human-like problem-solving** and enhance usability across multiple query types.

---

## 🎯 Objectives

- Implement a **RAG pipeline** to retrieve document-based knowledge.
- Use an **LLM** to synthesize user-friendly, natural language answers.
- Introduce **agentic workflows** that intelligently decide between:
  - Tools like calculators or dictionary APIs
  - The document-based RAG → LLM pipeline
- Build a **transparent interface** that showcases each internal step, offering insight into the assistant’s decision-making process.

---

## ✅ Features

- Vector-based retrieval using FAISS
- LLM answers with context-aware generation
- Agent decides whether to:
  - Evaluate a math expression
  - Answer a definition query
  - Perform standard RAG → LLM flow
- Logs all decision paths in `agent.log`
- Includes both **CLI** and **Web UI (Streamlit)**

---

## 📂 File Structure

```bash
multi_agent_rag_qa/
│
├── 📄 README.md            # Project overview and setup instructions
├── requirements.txt       # Python dependencies
│
├── docs/                  # Text documents (source knowledge base)
│
├── ingestion.py           # One-time vector creation from documents
├── retriever.py           # Handles query-time retrieval of top-k results
├── tools.py               # Agent logic and tool routing (calculator, dictionary, etc.)
├── log.py                 # Logging for agent decisions and actions
│
├── app.py                 # Web UI using Streamlit
├── cli.py                 # Optional CLI-based interface

---
