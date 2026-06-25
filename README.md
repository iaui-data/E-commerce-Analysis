# E-commerce User Behavior Analysis

- [Project Overview](#project-overview)
- [Key Technical Modules](#key-technical-modules)
- [Visualization Highlights](#visualization-highlights)
- [Business Impact](#business-impact)
- [Tech Stack](#tech-stack)
- [Future Improvements](#future-improvements)
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

## Future Improvements
- **Address Class Imbalance (Model Limitation):** The current XGBoost model yields an overall accuracy of 67%, but exhibits a low recall (0.02) for the minority class (purchasers). In future iterations, techniques like SMOTE (Synthetic Minority Over-sampling Technique) or adjusting the `scale_pos_weight` parameter will be implemented to improve the model's sensitivity in predicting actual purchase events.
- **Causal Inference via A/B Testing:** The current Chi-Square test relies on observational data, which proves correlation but not strict causation between `wishlist_count` and purchases. In a real-world business setting, I would design a randomized A/B test (e.g., hiding the wishlist feature for the control group) to rigorously quantify the incremental causal impact of this feature on the final conversion rate.
- **Data Enrichment:** Integrate user demographic data to further refine RFM segments.
- **Model Tuning:** Implement hyperparameter optimization (e.g., using GridSearchCV or Optuna) to improve prediction metrics.
