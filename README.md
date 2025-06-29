# 🧠 Federated Learning for Sepsis Prediction

This project implements a **Federated Learning** approach for predicting **Sepsis** using a distributed data setup. The system is built using Python and the [Flower](https://flower.dev) framework, simulating multiple hospitals (clients) collaboratively training a machine learning model without sharing their local data.

---

## 🚀 Project Overview

Sepsis is a life-threatening condition that arises when the body's response to infection causes injury to its own tissues and organs. Early prediction can significantly improve patient outcomes.

**Federated Learning (FL)** allows multiple institutions to collaboratively train machine learning models without sharing sensitive patient data, thus preserving privacy and complying with data regulations like HIPAA.

This project demonstrates a simple FL pipeline for predicting sepsis using local datasets at multiple clients, with server-side aggregation and prediction.

---

## 🧱 Architecture

- **Clients:** Represent hospitals with local patient data.
- **Server:** Coordinates model training and aggregates parameters.
- **Model:** `RandomForestClassifier` from `scikit-learn`.
- **Framework:** [Flower (FLWR)](https://flower.dev/)

```
        +-------------+       +-------------+       +-------------+
        |  Client 1   | <---> |             | <-->  |  Client 2   |
        |  (Hospital) |       |   Server    |       |  (Hospital) |
        +-------------+       +-------------+       +-------------+

         Train local model      Aggregate params     Train local model
         on local data         and send global      on local data
                                 model back
```

---

## 📦 Features

- Federated training using Flower framework
- Local training with `RandomForestClassifier`
- Secure and privacy-preserving learning
- Sepsis prediction model based on clinical data

---

## 🛠️ Tech Stack

- Python 3.x
- [scikit-learn](https://scikit-learn.org/)
- [Flower](https://flower.dev/)
- [UCI ML Repository Dataset #827](https://archive.ics.uci.edu/dataset/827)

---

## 📁 Folder Structure

```
federated_sepsis/
│
├── server.py            # Federated learning server
├── client1.py           # Client 1 code
├── client2.py           # Client 2 code
├── utils.py             # Shared utility functions
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── dataset/             # UCI Sepsis dataset (optional if fetched dynamically)
```

---

## 🧪 How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Server

```bash
python server.py
```

### 3. Run Clients (in separate terminals)

```bash
python client1.py
python client2.py
```

---

## 📊 Dataset

- **Source:** UCI Machine Learning Repository - Dataset #827
- **Content:** Vital signs, lab results, and patient metadata
- **Use:** For local training on each simulated hospital/client

---

## 📈 Metrics

- **Evaluation:** Accuracy, Precision, Recall
- **Cross-validation:** 80/20 Train-Test split per client
- **Model:** Random Forest for non-linear, high-dimensional clinical data

---

## ✅ Advantages of FL in Healthcare

- **Privacy-preserving** – raw data never leaves hospital
- **Compliance** – follows data protection regulations
- **Scalable** – works with any number of hospitals
- **Efficient** – enables global model improvement with local data

---

## ⚠️ Limitations

- Simulated clients (not deployed across real devices)
- Requires stable connectivity
- Not real-time or production-deployed

---

## 📚 References

- Flower Framework: https://flower.dev
- UCI Dataset: https://archive.ics.uci.edu/dataset/827
- Research on Sepsis Prediction: [PubMed](https://pubmed.ncbi.nlm.nih.gov/)

---

## 📬 Contact

Maintained by **Balaji K**  
If you use this project in your research, please give appropriate credit.

---

## 📄 License

MIT License - see `LICENSE` file for details.
