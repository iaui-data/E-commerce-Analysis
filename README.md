# E-commerce User Behavior Analysis

## Project Overview
This project provides a comprehensive analysis of e-commerce user behavior data. The goal is to identify bottlenecks in the user journey, segment high-value users, and predict purchase intent using machine learning models.

## Key Technical Modules
- **Funnel Analysis:** Identified a 7.15% conversion rate from view to purchase, highlighting the "Cart to Purchase" stage as the primary bottleneck.
- **RFM Customer Segmentation:** Performed RFM modeling to profile high-value user segments.
- **Purchase Intent Prediction:** Built an XGBoost classifier to predict user purchases. Achieved a 67% accuracy rate and identified `wishlist_count` as a critical predictive feature.

## Visualization Highlights

**1. Conversion Funnel**
![Conversion Funnel](output_3_0.png)

**2. Feature Importance (XGBoost)**
![Feature Importance](output_5_1.png)

**3. Activity vs. Revenue Correlation**
![Correlation](output_6_0.png)

## Business Impact
The analysis provides actionable insights for:
- Optimizing conversion pathways at the checkout stage.
- Targeted marketing strategies for different RFM segments.
- Resource allocation based on high-impact predictive features.

## Tech Stack
- **Languages:** Python
- **Libraries:** Pandas, Scikit-learn, XGBoost, Plotly, Seaborn
