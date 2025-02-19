# Hallucination Detection in Llama-2-7B using Ensemble Modelling

## Author
**Gayatri Kharche**  
Email: gkharche@ucsd.edu  

## Overview
This project aims to detect hallucinations in responses generated by Llama-2-7B, where the model produces false or fabricated information. Using the **FEVER dataset**, classification models (**DistilBERT, BERT, RoBERTa, ALBERT**) were trained to distinguish between truthful and fabricated answers. Performance was evaluated on both **imbalanced and balanced datasets**, with enhancements through **ensemble learning**.

## Tasks Completed
- ✅ **Collected and preprocessed the FEVER dataset**  
- ✅ **Generated Llama2 responses for evaluation**  
- ✅ **Trained classification models (DistilBERT, BERT, RoBERTa, ALBERT)**  
- ✅ **Evaluated performance on balanced and imbalanced datasets**  
- ✅ **Analyzed the impact of Llama-generated answers on classification**  
- ✅ **Optimized model performance beyond baseline**  
- ✅ **Enhanced results using ensemble learning**  
- ⚠ **Tried even better models (partially done due to computational limits)**  

## Dataset
**FEVER (Fact Extraction and Verification) Dataset**  
- Contains **311,431 claims** paired with evidence from Wikipedia.  
- Labels:
  - **SUPPORTS**: The claim is true based on evidence.  
  - **REFUTES**: The claim is false based on evidence.  
  - **NOT ENOUGH INFO**: The evidence is insufficient to verify the claim.  
- **Preprocessing**: Tokenization, label encoding, balancing for equal class distribution.

## Models Used
- **Baseline:** MobileBERT (performed poorly, ~26.7% accuracy)  
- **Fine-tuned models:**  
  - **DistilBERT**
  - **BERT**
  - **RoBERTa**
  - **ALBERT**  
- **Ensemble Model:** Combined multiple models for improved accuracy.  

## Llama-2-7B Hallucination Testing
- Generated claims using **Llama-2-7B Chat**  
- Example:
  - **Prompt:** "Who won the FIFA World Cup in 2022?"  
  - **Llama-2 Output:** "The FIFA World Cup was won by the France National team." (Incorrect)  
  - **Classifier Prediction:** REFUTES (Detected as hallucination)  

## Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/hallucination-detection.git
   cd hallucination-detection

