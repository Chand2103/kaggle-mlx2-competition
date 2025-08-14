# Music Popularity Prediction using Ensemble Learning

## Overview
This project presents multiple machine learning approaches to predict music popularity scores (0-100) using real-world music data from the mlX 2.0 Regression Challenge. Our team explored various methodologies, with the **stacked ensemble approach achieving the best performance with an RMSE of 9.0035**.

## Dataset Description

### Challenge Objective
Predict continuous popularity scores for music tracks using audio features, artist statistics, and track metadata. The goal is to analyze trends in danceability, energy, valence, and other musical characteristics to forecast listener reception before songs hit the charts.

### Data Structure
The dataset contains real Billboard-tracked tracks with the following key components:

#### Core Features
- **Track Identification**: `track_identifier`, `creator_collective`, `publication_timestamp`
- **Audio Features** (for tracks 0-2):
  - `rhythmic_cohesion_[0-2]`: Danceability (0-1)
  - `intensity_index_[0-2]`: Energy level (0-1)
  - `harmonic_scale_[0-2]`: Musical key (0-11)
  - `tonal_mode_[0-2]`: Modality (0=minor, 1=major)
  - `organic_texture_[0-2]`: Acousticness (0-1)
  - `beat_frequency_[0-2]`: Tempo in BPM
  - `time_signature_[0-2]`: Meter (e.g., 3=3/4, 4=4/4)
  - `duration_ms_[0-2]`: Duration in milliseconds

#### Derived Metrics
- `emotional_charge_[0-2]`: Valence × Energy product
- `groove_efficiency_[0-2]`: Energy/Danceability ratio
- `organic_immersion_[0-2]`: Acousticness × Duration
- `duration_consistency`: Standard deviation of track durations
- `tempo_volatility`: BPM range across tracks
- `key_variety`: Unique keys among first 3 tracks

#### Contextual Features
- `album_name_length`: Character length of track title
- `artist_count`: Number of credited artists
- `weekday_of_release`: Day of week
- `season_of_release`: Seasonal quarter
- `lunar_phase`: Moon phase at release

### Target Variable
- `target`: Continuous popularity score (0-100, higher values indicate more popular tracks)

## Team Approaches

### Best Performing Approach: Stacked Ensemble (RMSE: 9.0035)
**Location**: `6 model ensemble/9.0035.ipynb`

#### Ensemble Learning Strategy
The winning solution employs a **stacked ensemble meta-model** combining six tree-based regressors:

1. **XGBoost** - Extreme Gradient Boosting
2. **LightGBM** - Light Gradient Boosting Machine
3. **Extra Trees** - Extremely Randomized Trees
4. **Histogram GB** - Histogram-based Gradient Boosting
5. **Random Forest** - Ensemble of decision trees
6. **CatBoost** - Categorical boosting

#### Key Features
- **Hyperparameter Tuning**: Utilized Optuna for automated optimization
- **Cross-Validation**: Robust validation strategies
- **Feature Engineering**: Advanced feature creation and missing value handling
- **Model Stacking**: Meta-learning techniques for model combination
- **SHAP Analysis**: Comprehensive interpretability analysis

### Alternative Approaches Explored

#### 1. XGBoost Single Model Approach
**Location**: `notebooks-chandupa/notebook1.ipynb`
- Focused on XGBoost with feature engineering
- Implemented KNN imputation for missing values
- Used PCA for dimensionality reduction

#### 2. CatBoost Implementation
**Location**: `notebooks-chandupa/notebook4.ipynb`
- Leveraged CatBoost's categorical handling capabilities
- Implemented custom feature engineering
- Explored different hyperparameter configurations

#### 3. Traditional Ensemble Methods
**Location**: `notebooks-chandupa/`
- Combined multiple base models (Random Forest, Extra Trees)
- Implemented voting and averaging strategies
- Explored different preprocessing techniques

#### 4. Advanced Feature Engineering Approach
**Location**: `notebooks-chandupa/others code/main.ipynb`
- Focused on sophisticated feature creation
- Implemented domain-specific transformations
- Explored temporal and contextual features

## Performance Comparison

| Approach | RMSE Score | Key Features |
|----------|------------|--------------|
| **Stacked Ensemble** | **9.0035** | 6-model combination, Optuna tuning, SHAP analysis |
| XGBoost Single | ~9.5-10.0 | KNN imputation, PCA reduction |
| CatBoost | ~9.8-10.2 | Categorical handling, custom features |
| Traditional Ensemble | ~10.0-10.5 | Voting strategies, basic preprocessing |
| Feature Engineering | ~10.2-10.8 | Advanced transformations, domain knowledge |

## Feature Importance Analysis
Our ensemble model revealed the most influential features for predicting music popularity using SHAP (SHapley Additive exPlanations) analysis:

![Feature Importance](6%20model%20ensemble/Screenshot%202025-08-14%20010640.png)

*SHAP-based feature importance analysis showing the most critical musical characteristics that influence popularity scores. The analysis reveals which audio features, derived metrics, and contextual factors have the strongest predictive power.*

## Technical Implementation

### Model Optimization Techniques
- **Hyperparameter Tuning**: Optuna for automated optimization
- **Cross-Validation**: Stratified k-fold validation
- **Feature Engineering**: Derived features, missing value handling
- **Model Stacking**: Meta-learning for ensemble combination
- **Interpretability**: SHAP analysis for model understanding

### Data Preprocessing
- **Missing Value Handling**: KNN imputation, simple imputation
- **Feature Scaling**: StandardScaler, RobustScaler
- **Dimensionality Reduction**: PCA for feature compression
- **Categorical Encoding**: One-hot encoding, label encoding

```

## Key Findings

### Ensemble Learning Insights
- **Model Diversity**: Combining different algorithms improved robustness
- **Feature Importance**: Audio characteristics strongly influence popularity
- **Temporal Patterns**: Release timing affects popularity scores
- **Artist Factors**: Collaboration patterns impact track success

### Comparative Analysis
- **Single Models**: XGBoost and CatBoost showed strong individual performance
- **Ensemble Benefits**: Stacking multiple models consistently improved accuracy
- **Feature Engineering**: Domain-specific features enhanced model performance
- **Preprocessing Impact**: Proper handling of missing values was crucial

## Research Publication
The comprehensive findings from all approaches were published in the research paper **[Group 18 Research Paper.pdf](Research-paper/Group%2018%20Research%20Paper.pdf)**, which includes:
- Detailed methodology comparison
- Performance analysis across different approaches
- Insights into music popularity prediction
- Recommendations for future work

## Technologies Used
- **Python**: Primary programming language
- **XGBoost**: Gradient boosting framework
- **LightGBM**: Light gradient boosting machine
- **CatBoost**: Categorical boosting
- **Optuna**: Hyperparameter optimization
- **SHAP**: Model interpretability
- **Pandas/NumPy**: Data manipulation
- **Scikit-learn**: Machine learning utilities
- **KNN Imputation**: Missing value handling
