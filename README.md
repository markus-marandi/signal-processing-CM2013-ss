# signal-processing-CM2013-ss

# Automatic Sleep Scoring with Multi-Signal Processing

## Project Overview
This project aims to develop an automatic sleep scoring system that segments continuous sleep recordings into 30-second epochs and classifies each epoch into sleep stages according to AASM guidelines. The system leverages multiple physiological signals to improve classification accuracy through an iterative, phased approach.

## Project Objectives
- **Preprocess** raw biosignal data to remove artifacts.
- **Extract features** from time, frequency, and time-frequency domains.
- **Analyze features** statistically across different sleep stages.
- **Develop and optimize** machine learning models for sleep stage classification.
- **Optionally detect** sleep apnea episodes using SpO2 and respiration signals.

## Available Data
- **Data Files:**
  - 20 EDF files containing multiple biosignals:
    - **EEG:** 2 channels (brain activity)
    - **EOG:** 2 channels (left and right eye movements)
    - **EMG:** Muscle activity
    - **ECG:** Heart activity
    - **Body Position**
    - **SpO2:** Oxygen saturation
    - **Respiration:** Thoracic and abdominal signals
- **Annotations:**
  - XML files accompanying each EDF file, containing:
    - Sleep stage annotations (30-second epochs with start time and duration)
    - Signal quality markers
    - Event annotations (e.g., apnea episodes, desaturation events)
    - Sleep stage classifications (Wake, N1, N2, N3, REM)
- **File Access Recommendations:**
  - Use `edfread.m` for EDF signal extraction.
  - Use `xml_read.m` to parse XML annotations.
  - Convert loaded data to `.mat` format to improve access speed.

## Implementation Phases

### Phase 1: EEG-Only Pipeline
- **Preprocessing:**
  - Baseline drift removal.
  - Muscle noise filtering.
  - Power line interference removal.
  - Detection and handling of other artifacts.
- **Feature Extraction:**
  - **Time-Domain:** Mean, variance, skewness, kurtosis, zero-crossing rate, Hjorth parameters.
  - **Frequency-Domain:** Power in standard EEG bands (delta, theta, alpha, beta, gamma), spectral edge frequency, relative band powers.
  - **Time-Frequency:** Wavelet coefficients, spectral entropy.
- **Initial Classification:**
  - Use a simple k-NN classifier to establish baseline performance.
  - Evaluate results using confusion matrices.
  - Document limitations and areas for improvement.

### Phase 2: EEG + EOG Integration
- **EOG Preprocessing:**
  - Baseline correction and noise filtering.
- **EOG Feature Extraction:**
  - Detect sleep-related eye movements (optional).
  - Extract blink characteristics such as blink rate and movement density.
- **Combined Analysis:**
  - Merge EEG and EOG features.
  - Perform feature selection.
  - Compare performance against the EEG-only pipeline.

### Phase 3: Integration of Additional Signals (EMG, ECG, or Respiration)
- **EMG Preprocessing:**
  - Bandpass filtering.
  - Envelope extraction.
  - Artifact removal.
- **EMG Feature Extraction:**
  - Compute RMS amplitude, analyze frequency content, detect bursts, and evaluate muscle tone.
- **Final Integration & Optimization:**
  - Merge features from EEG, EOG, and EMG.
  - Apply feature selection techniques.
  - Fine-tune the final classification model.

## Technical Requirements
- **Performance Evaluation:**
  - Use MATLAB's Classification Learner app for training and model evaluation.
  - **Metrics:** Overall accuracy, per-stage accuracy, F-score, and confusion matrix.
- **Cross-Validation Techniques:**
  - Leave-one-subject-out cross-validation.
  - K-fold cross-validation for hyperparameter tuning.

## Implementation Guidelines
- **Iterative Development:**
  - Start with a simple model and incrementally add complexity.
  - Document performance improvements after each iteration.
  - Validate each enhancement quantitatively.
- **Code Organization:**
  - Modular functions for each processing component.
  - Clear documentation and version control.
  - Parameter configuration files for easy adjustments.
- **Optimization Strategies:**
  - Utilize feature selection methods.
  - Experiment with hyperparameter tuning.
  - Compare different classification algorithms.
  - Balance model performance with computational complexity.

## Timeline Recommendations
- **Weeks 1-3:** Implement the EEG-only pipeline (Phase 1).
- **Weeks 4-6:** Integrate EOG data and execute Phase 2 tasks.
- **Weeks 7-8:** Incorporate additional signals (EMG, ECG, Respiration) in Phase 3.
- **Weeks 9-10:** Optimize the full system and compile the final report.

## Report Guidelines
- **Structure:**
  - **Introduction (1 page):** Background and project objectives.
  - **Methods (2-3 pages):** Overview of preprocessing, feature extraction, and classification techniques.
  - **Results & Discussion (4-5 pages):** Key findings, performance improvements, and model comparisons.
  - **Conclusion (1 page):** Summary and suggestions for future work.
  - **References:** (Not included in page limit)
- **Constraints:**
  - Maximum report length: 15 pages.
  - Up to 12 combined figures and tables.
- **Focus Areas:**
  - Emphasize comparative analysis and the impact of iterative improvements.
  - Present clear performance metrics (accuracy, F-score, confusion matrix).
  - Use visual aids to illustrate performance improvements.
  - Document challenges and lessons learned throughout the project.

## Development Methodologies
- **Waterfall Development:**
  - Sequential, linear process with comprehensive documentation.
  - Less flexibility for adapting to unforeseen issues.
- **Iterative Development:**
  - Agile approach with repeated cycles of development and evaluation.
  - Early detection and resolution of issues.
  - Incremental performance gains through each iteration.
- **Iteration Plan:**
  - **Iteration 1:** Basic pipeline with minimal preprocessing and a simple k-NN classifier.
  - **Iteration 2:** Enhanced preprocessing with advanced artifact removal.
  - **Iteration 3:** Optimized feature selection (e.g., using PCA).
  - **Iteration 4:** Fine-tuning using classifiers such as SVM, Random Forest, and Neural Networks.
  - **Subsequent Iterations:** Continuous refinement based on validation results.

## Deliverables
- **Code Repository:**
  - Preprocessing scripts.
  - Feature extraction modules.
  - Classification model implementations.
  - Utility functions for data handling and conversions.
- **Final Report:**
  - Detailed methodology and results.
  - Comparative analysis of performance improvements.
  - Discussion of lessons learned and recommendations for future enhancements.

## References
- MATLAB Classification Learner App: [https://se.mathworks.com/help/stats/classification-learner-app.html](https://se.mathworks.com/help/stats/classification-learner-app.html)