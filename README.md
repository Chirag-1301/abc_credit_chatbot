# ABC Credit — Instant AI Loan Decision Engine & Chatbot 🚀

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.0.0-green.svg)](https://fastapi.tiangolo.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-0.9253_ROC--AUC-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![Groq LLM](https://img.shields.io/badge/Groq_LLM-Llama--3.1--8b-orange.svg)](https://groq.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade, real-time machine learning decision engine paired with an adaptive conversational AI assistant to deliver instant two-wheeler loan pre-approval decisions in **under 45 seconds** (SLA <= 1.5 mins) with sub-20ms model inference latency.

---

## 🌟 Key Features & Highlights

- **LightGBM Decision Engine**: Tuned via **Optuna Bayesian TPE** and calibrated using **Isotonic Regression** across 108,423 historical loan records (**0.9253 ROC-AUC**, **83.80% Accuracy**).
- **Pincode Prosperity Score**: Integrated Census-derived socio-economic index (1.0 - 100.0) mapped to 6-digit Pincodes, delivering a **+14.0% ROC-AUC uplift** over standard demographic baselines.
- **Adaptive Early Exit Cascade**: Parameters **X = 4 facts** (prerequisite financial fields) and **Y = [1.5%, 16.0%] PD** (confidence boundaries) for instant pre-approvals or rejections without unnecessary questions.
- **Strict Dual-Layer Architecture**: Generative Layer (**Groq LLM Llama-3.1-8b**) handles natural language chat and is strictly isolated from model risk probabilities.
- **100% PII Security & Masking**: Applicant identity fields (Name, Phone, Email, Pincode) are masked in memory and audit logs (`C*****`, `+91 ***** **210`, `56****`).
- **Data-Attributed Adverse Action Disclosures**: Adverse action reason codes (`R02-R10`) are strictly restricted to data provided by the user, translated by Groq LLM into warm, empathetic guidance.

---

## 📐 System Architecture

```
                  ┌──────────────────────────────────────────────────────────┐
                  │                 APPLICANT CHAT INTERFACE                 │
                  └────────────────────────────┬─────────────────────────────┘
                                               │ (User Message)
                                               ▼
                  ┌──────────────────────────────────────────────────────────┐
                  │              FASTAPI API SERVER (server.py)             │
                  └──────────────┬────────────────────────────┬──────────────┘
                                 │                            │
             (NLU Slot Extraction)                            │ (Structured Loan Facts)
                                 ▼                            ▼
                  ┌─────────────────────────────┐  ┌──────────────────────────┐
                  │    GROQ LLM GENERATIVE LAYER │  │  LIGHTGBM PREDICTIVE     │
                  │   (Llama-3.1-8b-instant)    │  │  DECISION ENGINE         │
                  │  * Isolated from model scores│  │  * 16.8ms Inference      │
                  │  * 4 Python Safety Gates    │  │  * 0.9253 ROC-AUC        │
                  └─────────────────────────────┘  └──────────┬───────────────┘
                                                              │
                                                              ▼
                                                   ┌──────────────────────────┐
                                                   │ ADAPTIVE EARLY EXIT      │
                                                   │ CASCADE (Params X & Y)   │
                                                   └──────────────────────────┘
```

---

## 📊 Empirical Model Performance & Metrics

Evaluated on `108,423` applications (`102,725` Approved vs `5,698` Declined):

| Metric | Score Achieved | Definition & Interpretation |
| :--- | :--- | :--- |
| **ROC-AUC Score** | **0.9253** | 92.53% separation power between approved and defaulted applicants. |
| **PR-AUC Score** | **0.3819** | Area under Precision-Recall curve (7.2x lift over random 0.0526 baseline). |
| **Declined Recall (0.0990 Thresh)** | **87.15%** | Catches 87.15% of all true defaults (4,966 / 5,698 caught). |
| **Declined Recall (0.0300 Cutoff)** | **96.81%** | Strict operational policy catching 96.81% defaults (only 182 missed). |
| **Approved Class Recall** | **83.61%** | Correctly approves 83.61% of creditworthy applicants automatically. |
| **Overall Accuracy** | **83.80%** | Overall correct classification rate across all applications. |
| **Inference Latency** | **16.8 ms** | Sub-20ms execution turnaround (<= 500ms SLA target). |

---

## ⚡ Quick Start Guide

### Prerequisites
- **Python 3.9+**
- **Groq API Key** (Set `GROQ_API_KEY` in `.env`)

### Installation & Server Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Chirag-1301/abc_credit_chatbot.git
   cd abc_credit_chatbot
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   PORT=8000
   ```

4. **Launch the FastAPI Server**:
   ```bash
   python server/server.py
   ```
   The server will start at `http://localhost:8000`.

5. **Access the Chatbot Application**:
   Open `public/index.html` in any browser or navigate to `http://localhost:8000`.

---

## 📁 Project Structure

```
abc_credit_chatbot/
├── ABC_Credit_Loan_Approval_Chatbot_Pipeline.ipynb  # End-to-end ML notebook & Optuna tuning
├── ABC_Credit_Loan_Approval_Executive_Deck.pptx     # McKinsey-style MBA Executive Deck
├── ABC_Credit_Loan_Decision_Engine_Report.md        # Comprehensive technical & business report
├── generate_deck.py                                # Python pptx presentation generator script
├── data/
│   ├── training_data_3_aug.csv                     # Historical dataset (108,423 records)
│   ├── Prosperity_Master_Lookup.csv               # 6-digit Pincode Prosperity Index
│   └── Vehicle_Catalog.csv                        # Two-wheeler specification catalogue
├── model/
│   ├── lgbm_model.joblib                          # Calibrated LightGBM model binary
│   └── model_metadata.json                        # Preprocessing schemas & risk maps
├── public/
│   ├── index.html                                 # Web Chatbot Frontend Interface
│   └── app.js                                     # Dynamic conversational UI logic
└── server/
    └── server.py                                  # FastAPI backend & Groq LLM integration
```

---

## 📜 License & Author

- **Author**: Chirag Khandelwal ([@Chirag-1301](https://github.com/Chirag-1301))
- **Institution**: Indian Institute of Management (IIM)
- **License**: MIT License
