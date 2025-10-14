# Applications of AI in Infosec - Jupyter Notebooks

This repository contains Jupyter notebooks implementing machine learning solutions for cybersecurity applications, based on the **Applications of AI in Infosec** learning module from [Hack The Box Academy](https://academy.hackthebox.com/module/details/292).

## 📚 Module Overview

The **Applications of AI in Infosec** module explores how artificial intelligence and machine learning techniques can be applied to solve real-world cybersecurity challenges. This comprehensive module covers three key areas where AI demonstrates significant value in information security.

## 🗂️ Repository Contents

### 1. **Network Anomaly Detection** (`network_anomaly_detection.ipynb`)
- **Objective**: Detect anomalous network traffic patterns using machine learning
- **Technique**: Random Forest Classifier for intrusion detection
- **Dataset**: NSL-KDD dataset for network intrusion detection
- **Key Features**:
  - Binary and multi-class classification approaches
  - Feature engineering and preprocessing
  - Model evaluation with comprehensive metrics
  - Attack type categorization (DoS, Probe, Privilege Escalation, Access)

### 2. **Malware Classification** (`malware_classification.ipynb`)
- **Objective**: Classify malware samples into families using deep learning
- **Technique**: Convolutional Neural Networks (CNN) with transfer learning
- **Dataset**: Malimg dataset (malware images)
- **Key Features**:
  - Malware binary-to-image conversion
  - Pre-trained ResNet50 architecture
  - Transfer learning with weight freezing
  - 25 malware family classification
  - Training visualization and performance analysis

### 3. **Spam Classification** (`Spam_Classification.ipynb`)
- **Objective**: Detect spam messages using natural language processing
- **Technique**: Multinomial Naive Bayes classifier
- **Dataset**: SMS Spam Collection dataset
- **Key Features**:
  - Text preprocessing and feature extraction
  - Bag-of-words model with n-grams
  - Hyperparameter tuning with GridSearchCV
  - Model persistence and evaluation

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- Jupyter Notebook
- Required Python packages (see individual notebooks for specific requirements)

### Installation
1. Clone this repository:
```bash
git clone <repository-url>
cd htb-module
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

### Running the Notebooks
Each notebook is self-contained and can be run independently:

1. **Network Anomaly Detection**: Download NSL-KDD dataset and run through the complete ML pipeline
2. **Malware Classification**: Download Malimg dataset and train a CNN classifier
3. **Spam Classification**: Process SMS data and build a spam detection model

## 📊 Key Learning Outcomes

After completing these notebooks, you will understand:

- **Machine Learning in Cybersecurity**: How ML algorithms can detect threats and anomalies
- **Feature Engineering**: Techniques for preparing security-relevant data
- **Model Evaluation**: Metrics and methods for assessing security ML models
- **Deep Learning Applications**: Using CNNs for malware analysis
- **Natural Language Processing**: Text analysis for security applications
- **Transfer Learning**: Leveraging pre-trained models for security tasks

## 🛡️ Security Applications

These implementations demonstrate practical AI applications in:

- **Intrusion Detection Systems (IDS)**: Automated network monitoring
- **Malware Analysis**: Rapid classification of malicious software
- **Email Security**: Spam and phishing detection
- **Threat Intelligence**: Automated threat classification and analysis

## 📈 Performance Highlights

- **Network Anomaly Detection**: Multi-class classification with weighted metrics
- **Malware Classification**: 88.54% accuracy on test dataset using transfer learning
- **Spam Classification**: High precision spam detection with optimized hyperparameters

## 🔧 Technical Stack

- **Machine Learning**: scikit-learn, PyTorch, torchvision
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Text Processing**: NLTK, CountVectorizer
- **Model Persistence**: joblib, TorchScript

## 📖 Educational Value

These notebooks serve as:
- **Learning Resources**: Step-by-step implementations with detailed explanations
- **Reference Implementations**: Production-ready code patterns for security ML
- **Research Starting Points**: Extensible frameworks for further experimentation
- **Best Practices**: Demonstrations of proper ML workflow in security contexts

## 📄 License

This project is for educational purposes. Please respect the terms of use for the datasets and Hack The Box Academy content.

## 🔗 References

- **Hack The Box Academy**: [Applications of AI in Infosec Module](https://academy.hackthebox.com/module/details/292)
- **NSL-KDD Dataset**: Network intrusion detection benchmark
- **Malimg Dataset**: Malware image classification dataset
- **SMS Spam Collection**: Text classification dataset

---

**Note**: These notebooks are based on educational content from Hack The Box Academy and are intended for learning purposes. Always ensure you have proper authorization before applying these techniques to real systems or data.
