# 🩻 Multi-Modal Medical Report Generation Pipeline

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![Model](https://img.shields.io/badge/LLM-BioGPT-green)
![Technique](https://img.shields.io/badge/Fine--Tuning-PEFT%2FLoRA-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **State-of-the-Art Radiology Reporting:** A novel deep learning framework that fuses Multi-View Chest X-Rays (Frontal & Lateral) using Swin Transformers and generates clinical reports via a fine-tuned BioGPT Large Language Model.

---

## 📖 Overview

In clinical practice, radiologists rely on both frontal (postero-anterior) and lateral projections to diagnose thoracic pathology accurately. However, most existing AI report generation systems rely on single-view analysis, leading to incomplete feature interpretation.

This project implements a **Two-Phase Deep Learning Pipeline** to address this limitation:
1.  **Vision Phase:** Extracts and fuses features from multi-view X-rays using a **Swin Transformer** backbone and a **Bidirectional Cross-Attention (BCA)** mechanism.
2.  **Language Phase:** Generates clinically coherent text using **BioGPT**, fine-tuned with **PEFT-LoRA** to ensure computational efficiency and diagnostic accuracy.

---

## 🚀 Key Features

* **Multi-View Fusion:** Implements a Bidirectional Cross-Attention module that allows dynamic contextual exchange between Frontal and Lateral views, capturing finer-grained spatial relationships.
* **Biomedical LLM Integration:** Utilizes Microsoft's **BioGPT**, a domain-specific language model pre-trained on biomedical corpora, ensuring high-quality medical terminology generation.
* **Parameter-Efficient Fine-Tuning:** Uses **LoRA (Low-Rank Adaptation)** to fine-tune the LLM while keeping the vision backbone frozen, significantly reducing computational cost without compromising adaptability.
* **Diagnostic Benchmarking:** The vision backbone is explicitly pre-trained on multi-label classification tasks to validate diagnostic competence before being used for text generation.

---

## 🏗️ Methodology & Architecture

The system follows a modular pipeline approach:

### Phase 1: The Vision Backbone
* **Encoder:** Swin Transformer (Hierarchical Vision Transformer).
* **Fusion:** Bidirectional Cross-Attention (BCA) fuses Frontal and Lateral features into a `Unified Patient Embedding`.
* **Validation:** Benchmarked using AUC-ROC on 14 thoracic findings (e.g., Cardiomegaly, Consolidation).

### Phase 2: Report Generation
* **Bridge:** A lightweight Visual Prefix Mapper (MLP) converts the unified visual embedding into soft prompts.
* **Decoder:** BioGPT generates the radiology report conditioned on the visual prompts.
* **Optimization:** Causal Language Modeling (CLM) loss optimized via PEFT-LoRA.

---

## 📊 Performance & Results

The model was trained and evaluated on the **Indiana University (IU) Chest X-ray Dataset**.

### 📝 Text Generation Metrics
The system achieved state-of-the-art results in semantic alignment:

| Metric | Score | Significance |
| :--- | :--- | :--- |
| **BERTScore-F1** | **0.9088** | Indicates high semantic similarity to expert radiologist reports. |
| **ROUGE-L** | **0.4176** | Shows strong structural sentence similarity. |
| **BLEU-4** | **0.1662** | Measures precision of generated n-grams. |

### 🩺 Diagnostic Accuracy (Vision Backbone)
Before text generation, the vision backbone demonstrated strong diagnostic capability:
* **Consolidation:** AUC ≈ 0.80.
* **Cardiomegaly / Edema:** AUC ≈ 0.50 - 0.55 (Moderate performance due to class imbalance).

---

## 🛠️ Installation & Usage

### Prerequisites
* Python 3.8+
* PyTorch 1.12+
* CUDA-enabled GPU (Recommended)

### Setup
```bash
# 1. Clone the repository
git clone [https://github.com/yourusername/medical-report-generation.git](https://github.com/yourusername/medical-report-generation.git)
cd medical-report-generation

# 2. Install dependencies
pip install -r requirements.txt
