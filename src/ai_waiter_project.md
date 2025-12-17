# AI Waiter: Agentic RAG Powered Restaurant Assistant

An **Agentic Retrieval Augmented Generation (RAG)** system that behaves like a real restaurant waiter by answering menu-related questions **only from verified menu data**, not model memory.

---

## Overview

**AI Waiter** is an intelligent assistant that understands restaurant menus stored in structured files (Excel), retrieves the most relevant information using vector search, and generates grounded, accurate responses using an agent-driven workflow.

Unlike traditional RAG systems, this project uses **Agentic RAG** allowing the system to reason about *when and how* to retrieve information before answering.

---

## Problem Statement

Most conversational assistants:

- hallucinate menu items
- show outdated prices
- fail when menus change
- blindly answer without verifying sources

**Restaurants need an assistant that is:**

- accurate
- grounded in the latest menu
- adaptable to updates
- conversational yet reliable

---

## Solution

AI Waiter solves this using an **Agentic RAG pipeline**:

1. User asks a question
2. Agent decides whether retrieval is needed
3. Menu data is retrieved from a vector database
4. Context is evaluated
5. Answer is generated strictly from retrieved data

This ensures **no** **hallucination** and **menu-faithful responses**.

---
