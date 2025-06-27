# 🧠 Federated Learning for Sepsis Prediction

This project implements a **Federated Learning (FL)** system to predict the onset of **sepsis** using distributed client datasets. The key goal is to maintain data privacy while building an accurate predictive model without transferring sensitive patient information.

---

## 📌 Project Highlights

- **Model Type**: Logistic Regression  
- **Frameworks**: Flower (Federated Learning), Flask (UI)  
- **Objective**: Early prediction of sepsis using decentralized hospital/patient data  
- **Privacy First**: No raw data sharing; only model parameters exchanged  
- **UI Capabilities**:
  - Launch FL server and clients
  - Real-time log streaming
  - Display client-side model coefficients
  - Input sample data for prediction
  - Show final prediction based on aggregated server model

---

## 🧬 Architecture

