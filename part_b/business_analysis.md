# Promotion Effectiveness at a Fashion Retail Chain

## B1. Problem Formulation

### (a) ML Problem Definition

Target Variable:
items_sold (number of items sold per store per month)

Input Features (Candidate Features):
- Store attributes: store_size, location_type, competition_density
- Promotion details: promotion_type
- Time features: month, year, is_weekend, is_festival
- Customer behavior: footfall, demographics
- Historical performance: past sales

Type of ML Problem:
Supervised Regression Problem

Justification:
The objective is to predict a continuous numerical value (items sold) using historical labeled data. Hence, this is a supervised learning task, and regression is appropriate.


### (b) Why Items Sold Instead of Revenue

Revenue can be misleading because:
- Promotions like discounts reduce price, affecting revenue
- High revenue does not always indicate high customer response

Advantages of using items_sold:
- Directly measures promotion effectiveness
- Reflects actual customer demand
- Independent of pricing strategies

Broader Principle:
The target variable must align directly with the business objective. Choosing an incorrect target leads to misleading models and poor decisions.


### (c) Alternative Modelling Strategy

Instead of a single global model, use:

Cluster-based or Hierarchical Modeling
- Group stores based on location_type or behavior
- Train separate models for each group
OR
- Use hierarchical (multi-level) models

Justification:
Different stores respond differently to promotions, so this approach captures local variations and improves accuracy.


## B2. Data and EDA Strategy

### (a) Data Joining and Dataset Design

Tables:
- Transactions
- Store Attributes
- Promotion Details
- Calendar

Joining Strategy:
- Join using store_id and transaction_date

Final Dataset Grain:
One row = one store per month

Aggregations:
- Total items_sold per store per month
- Average footfall
- Promotion applied in that month
- Number of transactions
- Weekend and festival indicators


### (b) EDA Strategy

1. Sales vs Promotion Type (Boxplot)
   - Identify which promotions perform best
   - Helps in feature selection

2. Time Series Analysis
   - Detect seasonality and trends
   - Useful for time-based features

3. Correlation Heatmap
   - Identify relationships between variables
   - Helps remove redundant features

4. Sales by Store Type (Bar Chart)
   - Compare urban, semi-urban, and rural performance
   - Helps in segmentation strategy


### (c) Handling Imbalance

Problem:
80% of transactions occur without promotions

Effects:
- Model becomes biased toward no-promotion cases
- Poor learning of promotion impact

Solutions:
- Oversampling promotion data
- Stratified sampling
- Using class weights or balanced loss
- Separate models for promotion vs no promotion


## B3. Model Evaluation and Deployment

### (a) Train-Test Split and Metrics

Split Strategy:
Time-based split
- Train on first 2 years
- Test on last 1 year

Why Random Split is Inappropriate:
- Breaks temporal dependency
- Causes data leakage

Evaluation Metrics:
- MAE (Mean Absolute Error): average prediction error in units sold
- RMSE (Root Mean Squared Error): penalizes larger errors
- R² Score: measures how well the model explains variance


### (b) Explaining Model Decisions

Use Feature Importance or SHAP values

Example:
- December: festival season → loyalty points perform better
- March: lower demand → discounts are more effective

Communication:
- Show feature importance plots
- Explain seasonal influence
- Translate insights into simple business terms


### (c) Deployment Pipeline

1. Model Saving:
   joblib.dump(model, 'model.pkl')

2. Monthly Prediction Pipeline:
- Collect new data (store, calendar, promotions)
- Apply same preprocessing steps
- Generate predictions

3. Recommendation System:
- Predict items_sold for each promotion type
- Select promotion with highest predicted sales

4. Monitoring:
- Track prediction vs actual error
- Monitor feature drift
- Observe performance degradation

Retraining Trigger:
- Significant increase in error
- Changes in seasonal or customer behavior