# SMLP: NLP, LLM, RAG, and Agent Capabilities

SMLP (Symbolic Machine Learning Prover) is a model exploration, verification, synthesis, and optimization framework for machine-learning and statistical models. It combines machine learning with symbolic reasoning to support tasks such as model analysis, verification, root-cause analysis, synthesis, optimization, and robust exploration.

In addition to its core symbolic-ML capabilities, SMLP provides an extended set of NLP, LLM, Retrieval-Augmented Generation (RAG), evaluation, and agent capabilities.

The capabilities described below are implemented in the SMLP NLP/LLM extensions and are documented in detail in the **SMLP Extended Manual, Chapters 14–24**.

> **Documentation:** See the [SMLP Extended Manual](https://raw.githubusercontent.com/SMLP-Systems/smlp/nlp_text.rebased/doc/smlp_manual_extended.pdf) for detailed descriptions, configuration parameters, supported workflows, examples, and CLI reference.

---

## NLP and Text Processing

SMLP provides a configurable NLP preprocessing module based on **spaCy**. The NLP pipeline can be used as a preprocessing stage for text vectorization, classification, regression, retrieval, and other downstream ML workflows.

The NLP module supports, among other capabilities:

- Lemmatization
- Part-of-speech (POS) tagging
- Named Entity Recognition (NER)
- Sentence segmentation
- Syntactic parsing
- Morphological processing
- Token filtering
- Rule-based linguistic processing
- spaCy-based token/vector processing

The preprocessing pipeline is configurable so that relevant spaCy components can be enabled according to the application.

The NLP functionality integrates with the SMLP text-processing and RAG workflows.

See **Chapter 14** of the extended manual for details.

---

## Text Feature Engineering

SMLP provides a `SmlpText` module for converting textual data into numerical features that can be used by conventional ML models and SMLP analysis workflows.

Supported text representations include:

- Bag-of-Words (BoW)
- TF-IDF
- Word2Vec
- GloVe
- CBOW
- Skip-gram / fastText-based embeddings

Text features can be integrated into:

- Classification
- Regression
- Root-cause analysis
- Subgroup discovery
- Anomaly and event/log analysis
- Other SMLP model-exploration workflows

This allows textual information to be treated as a source of ML features alongside conventional numerical and categorical data.

See **Chapter 15** of the extended manual.

---

## Text Classification, Regression, and Subgroup Discovery

Text data can be used directly in SMLP training and analysis workflows.

SMLP can:

1. preprocess textual inputs;
2. convert the text into numerical features;
3. combine the resulting features with other data;
4. apply standard SMLP ML workflows.

Text data can also be analyzed using SMLP's subgroup-discovery capabilities.

This enables applications such as:

- Text classification
- Text regression
- Analysis of textual event or trace data
- Identification of subgroups associated with particular outcomes
- Text-based root-cause analysis

See **Chapter 16** of the extended manual.

---

# LLM Training from Scratch

SMLP provides support for training transformer-based language models from scratch.

Two levels of functionality are available:

### Experimental symbolic trainer

`SmlpGenerate` provides a lightweight experimental mechanism based on SMLP's custom Transformer and language-model implementations. It is intended primarily for experimentation and demonstrations involving symbolic/model-training concepts.

### Hugging Face-compatible training

`ScratchTrainer` provides a more general training interface based on the Hugging Face ecosystem.

It supports:

- Training tokenizers from raw text
- Byte-Pair Encoding (BPE)
- User-provided or automatically trained tokenizers
- Dataset preparation from raw text
- Block-based language-model training
- Hugging Face-compatible language-model classes
- Training, evaluation, checkpointing, and model persistence
- Text generation after training

The `SmlpScratch` integration exposes scratch LLM training through the main SMLP workflow.

The scratch-training functionality is intended to provide an integrated SMLP workflow for experimenting with domain-specific language models rather than replacing large-scale industrial LLM training infrastructure.

See **Chapter 17** of the extended manual.

---

# Retrieval-Augmented Generation (RAG)

SMLP provides Retrieval-Augmented Generation capabilities for document-based question answering, summarization, and natural-language understanding.

Two complementary RAG approaches are supported:

- **Hugging Face RAG (HF-RAG)**
- **LangChain RAG (LC-RAG)**

Both approaches allow SMLP to work with external document collections and generate responses using retrieved context.

## Hugging Face RAG

HF-RAG is based on Hugging Face's RAG architecture and supports:

- Retrieval from document collections
- Question answering
- Generation using retrieved passages
- Fine-tuning of the RAG model
- Custom QA-oriented datasets
- FAISS and cosine-similarity retrieval
- Saving and reusing trained RAG models and artifacts

HF-RAG provides tighter integration between retrieval and model training, but is correspondingly more resource-intensive.

## LangChain RAG

LC-RAG provides a more modular retrieval/generation architecture based on **LangChain**.

It supports:

- Document ingestion and chunking
- Embedding generation
- Configurable retrieval
- Configurable top-k retrieval
- Prompt templates
- External LLM endpoints
- Local LLM inference through systems such as Ollama
- OpenAI-based LLM endpoints
- Saving and reusing RAG artifacts

Unlike HF-RAG, LC-RAG does not retrain the underlying language model. Its main focus is retrieval, prompting, and integration with an external generation model.

## Supported document formats

SMLP RAG workflows support:

- PDF
- JSON
- CSV

PDF processing can use several parsing strategies, including:

- LangChain PDF loaders
- Hugging Face-oriented text processing
- Unstructured
- OCR/Tesseract-based extraction

The choice of parsing strategy can be important for documents containing complex layouts, tables, scanned pages, or non-linear structures.

## Retrieval backends

RAG workflows support retrieval based on:

- FAISS
- Cosine similarity

The retrieval configuration, passage segmentation, embeddings, and prompt construction can be controlled through SMLP configuration parameters.

## Prompting

LC-RAG supports configurable prompt styles, including:

- `document_focused`
- `chain_of_thought`
- `extractive`
- `strict_qa`
- `summarizer`

This allows the RAG workflow to be adapted to different application requirements.

See **Chapter 18** of the extended manual for the complete RAG documentation.

---

# LLM Fine-Tuning

SMLP provides an integrated workflow for fine-tuning pretrained language models on custom datasets.

Supported fine-tuning tasks include:

- Text generation
- Generative question answering
- Extractive question answering
- Summarization

The supported model families include:

- Decoder-only / causal language models
- Encoder-decoder / sequence-to-sequence models
- Encoder-only models for extractive QA

Examples of supported model families include TinyLlama, LLaMA-family models, Falcon, T5/FLAN-T5, BERT, DistilBERT, and BART, subject to the compatibility of the selected task and model architecture.

Fine-tuning datasets can represent:

- Text-generation examples
- Question/context/answer examples
- Extractive QA examples
- Input/summary pairs

The fine-tuning workflow includes training and evaluation and supports persistence and reuse of trained models.

SMLP integrates fine-tuning into the same overall command-line workflow used by its other ML and LLM capabilities.

The implementation uses the **Hugging Face Transformers ecosystem** and associated model/dataset tooling.

See **Chapter 19** of the extended manual.

---

# LLM-as-a-Judge Evaluation

SMLP provides an LLM-as-a-Judge evaluation framework for assessing generated text across multiple LLM workflows.

The same evaluation mechanism can be applied to:

- Training from scratch
- Fine-tuning
- RAG

The judge evaluates generated answers using configurable criteria including:

- Groundedness
- Hallucination
- Overall quality

The default evaluation produces a structured result containing:

- Groundedness result
- Hallucination result
- Numerical quality score
- Explanation

The judge prompt is configurable and can be replaced with a user-provided prompt template.

SMLP supports both:

- Local Hugging Face judge models
- OpenAI-based judge models

Judge results are stored in structured JSON form with summary and per-example details, enabling:

- Evaluation reporting
- Regression testing
- Comparison of models
- Comparison of training runs
- Downstream analysis

LLM-as-a-Judge complements conventional metrics such as loss and accuracy rather than replacing them. It is particularly useful when semantic quality, groundedness, hallucination, or multiple valid answers make exact-match metrics insufficient.

The evaluation interface is designed so that additional evaluation strategies can be integrated in the future.

See **Chapter 20** of the extended manual.

---

# SMLP Agent

SMLP provides an LLM-powered agent interface that translates natural-language requests into structured SMLP commands and executes the corresponding workflows.

The SMLP Agent can interpret requests involving capabilities such as:

- Model training
- Fine-tuning
- Inference
- RAG
- Evaluation
- Other SMLP workflows

The agent combines:

- An LLM for natural-language interpretation and reasoning
- Structured SMLP tools
- Explicit workflow/control logic

Rather than relying entirely on unconstrained autonomous tool use, the current agent uses structured execution mechanisms to improve reproducibility, control, and debuggability.

The agent can:

- Interpret natural-language SMLP requests
- Extract task parameters
- Generate structured SMLP commands/specifications
- Select appropriate SMLP capabilities
- Execute SMLP workflows
- Return execution results

The agent exposes a programmatic API based on **FastAPI**, allowing integration with applications and services.

The API includes functionality for:

- Natural-language task execution
- Specification preview without execution
- Structured task interaction
- Chat-oriented responses
- Dynamic prompt configuration
- Execution logging

The agent can use local LLMs through **Ollama** as well as API-based LLM providers such as **OpenAI**, subject to the configured backend.

See **Chapter 21** of the extended manual.

---

# SMLP Chatbot

SMLP also provides a web-based chatbot interface for interacting with the SMLP Agent using natural-language instructions.

The chatbot provides:

- Natural-language SMLP task specification
- Execution through the SMLP Agent API
- Session history
- Display of execution results
- Dynamic few-shot prompt configuration
- Upload or selection of custom prompt templates

The chatbot is implemented using **Streamlit** and communicates with the SMLP Agent through its API.

This provides a convenient interactive interface for users who prefer conversational interaction over direct command-line invocation.

See **Chapter 22** of the extended manual.

---

# Model Context Protocol (MCP) Support

SMLP provides integration with the **Model Context Protocol (MCP)**, allowing external MCP-compatible clients and applications to invoke SMLP functionality through structured, schema-defined tools.

The MCP integration consists of:

- An SMLP MCP server
- An SMLP MCP client

The server exposes SMLP functionality as MCP tools with defined input and output schemas.

The current implementation provides an MCP tool for running SMLP with specified parameters.

The MCP integration is based on **FastMCP**.

The current implementation uses local **stdio transport**, allowing an MCP client and server to communicate through standard input/output.

This integration provides a path for SMLP to participate as a structured ML/model-exploration tool within broader MCP-enabled AI and agent ecosystems.

See **Chapter 23** of the extended manual.

---

# SMLP RL Agent

SMLP includes an experimental reinforcement-learning-based extension of the SMLP Agent.

The **SMLP RL Agent** is designed to improve natural-language-to-SMLP-command translation using feedback from user corrections and ratings.

Instead of modifying the underlying LLM weights during the primary learning loop, the current approach learns how to select effective few-shot examples for the LLM.

The current RL mechanism uses a contextual-bandit approach based on **Upper Confidence Bound (UCB)** selection.

Conceptually, the workflow is:

```text
User query
    ↓
Select few-shot examples
    ↓
Construct prompt
    ↓
LLM generates SMLP command
    ↓
User correction / feedback
    ↓
Reward
    ↓
Update example ranking
    ↓
Improved example selection for future queries
```

These capabilities can also be accessed through the SMLP Agent and, where applicable, through the MCP interface.

This creates an integrated workflow spanning:

- Conventional ML
- Symbolic model exploration
- NLP
- Text feature engineering
- LLM training
- LLM fine-tuning
- Retrieval-Augmented Generation
- LLM-based evaluation
- Natural-language task specification
- Agent-based execution
- MCP-based integration

---

# Main Software Ecosystem

The SMLP NLP/LLM extensions build on widely used open-source and commercial ML/LLM technologies, including:

| Component | Role in SMLP |
|---|---|
| **spaCy** | NLP preprocessing, linguistic analysis, token processing |
| **gensim** | Text embeddings such as GloVe and related embedding workflows |
| **Hugging Face Transformers** | Transformer models, training, fine-tuning, generation |
| **Hugging Face Datasets** | Dataset preparation and processing |
| **Hugging Face Tokenizers** | Tokenizer training and processing |
| **Hugging Face Trainer** | LLM training and fine-tuning workflows |
| **LangChain** | Modular RAG pipelines, document loading, retrieval, prompting |
| **FAISS** | Vector retrieval/indexing |
| **Unstructured** | Parsing complex documents and structured/unstructured layouts |
| **Tesseract OCR** | OCR-based processing of scanned documents |
| **Ollama** | Local LLM serving/inference |
| **OpenAI APIs** | API-based LLM inference and evaluation |
| **LangGraph** | Structured agent/task-graph execution |
| **FastAPI** | SMLP Agent REST API |
| **Streamlit** | Interactive SMLP chatbot interface |
| **FastMCP** | Model Context Protocol integration |

The exact set of dependencies required depends on which SMLP capabilities are used.

---

# Design Philosophy

The NLP/LLM extensions are designed as an extension of SMLP's existing model-exploration philosophy rather than as a standalone generic LLM framework.

The emphasis is on integrating modern language-model capabilities with SMLP's existing strengths in:

- Model exploration
- Verification
- Root-cause analysis
- Synthesis
- Optimization
- Explainability
- Reproducible ML workflows

In particular, the SMLP Agent follows a structured approach in which the LLM provides natural-language understanding and task interpretation while explicit SMLP workflows provide controlled execution.

This allows LLM capabilities to be used as part of a broader model-analysis and reasoning environment rather than treating the LLM itself as the entire system.

---

# Documentation

The complete documentation for the capabilities described above is provided in the **SMLP Extended Manual**:

**Chapters 14–16**

- NLP preprocessing
- Text feature engineering
- Text classification, regression, and subgroup discovery

**Chapter 17**

- LLM training from scratch

**Chapter 18**

- Retrieval-Augmented Generation
- Hugging Face RAG
- LangChain RAG

**Chapter 19**

- LLM fine-tuning

**Chapter 20**

- LLM-as-a-Judge evaluation

**Chapter 21**

- SMLP Agent and API

**Chapter 22**

- SMLP Chatbot

**Chapter 23**

- Model Context Protocol (MCP) support

**Chapter 24**

- SMLP RL Agent

For complete configuration, CLI parameters, supported models, data formats, examples, limitations, and workflow details, consult the extended manual rather than relying on this overview.

---

# Related SMLP Documentation

- SMLP repository: [https://github.com/SMLP-Systems/smlp](https://github.com/SMLP-Systems/smlp)
- Extended SMLP Manual: [https://raw.githubusercontent.com/SMLP-Systems/smlp/nlp_text.rebased/doc/smlp_manual_extended.pdf](https://raw.githubusercontent.com/SMLP-Systems/smlp/nlp_text.rebased/doc/smlp_manual_extended.pdf)
