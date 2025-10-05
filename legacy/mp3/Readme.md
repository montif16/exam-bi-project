To run the code. Open the program folder with an anaconda environment.

Next in terminal
conda install pandas numpy scikit-learn plotly


# ---------- 1) Step 1 loading the data ----------
We loaded the IBM HR dataset (1470 rows × 35 columns). We removed constant/ID columns (EmployeeCount, Over18, StandardHours, EmployeeNumber). Attrition is highly imbalanced (1233 stay vs 237 leave). No missing values were found. We identified numeric features (e.g. Age, DistanceFromHome, MonthlyRate) and categorical features (e.g. Department, Gender, JobRole, OverTime).


# ---------- 2) Step 2 Employee income ----------
Our regression model predicts MonthlyIncome with an R² of 0.938 and an RMSE of ~1167, meaning it explains most of the salary variation and has a relatively low average error.

# ---------- 3) Step 3 Employee attrition  ----------
Our logistic regression model predicts attrition with 86% overall accuracy, but recall for the attrition class is only 34%, showing the challenge of class imbalance. The model is very reliable at predicting who will stay, but less so at identifying who will quit.

# -------- 3.5) Step 3.5 Handling Class Imbalance  ----------
In Step 3, our logistic regression model reached 86% accuracy overall, but only **0.34 recall** for the minority class (quitters). This meant the model missed many employees who actually left.

Since HR is more concerned with catching employees at risk of leaving, we retrained with `class_weight="balanced"`. This improved recall for quitters to **0.64**, almost double, while reducing overall accuracy to 76%. 

- True positives (quitters correctly identified) rose from 16% → 30%
- False negatives (quitters missed) dropped from 31% → 17%
- ROC-AUC = 0.803, PR-AUC = 0.559 (good indicators for imbalanced data)

**Conclusion:** Step 3.5 prioritises recall for attrition, making it more practical for HR decision-making, even if it means more false alarms.
Code at the bottom can be uncommented. It will catch more quitters (81%) But lower accuracy.

# -------- 4) Step 4 Clustering  ----------

We applied KMeans clustering with k = 2…10 and evaluated with silhouette score.

| k (clusters) | Silhouette Score |
|--------------|------------------|
| 2            | 0.114            |
| 3            | 0.107            |
| 4            | 0.069            |
| 5            | 0.067            |
| 6            | 0.061            |
| 7            | 0.055            |
| 8            | 0.057            |
| 9            | 0.048            |
| 10           | 0.048            |

- The best score was with **k=2** (0.114).  
- Cluster distribution: 958 employees vs 512 employees.  
- Since the scores are low (<0.25), this indicates the dataset does not separate cleanly into clusters — employees are quite diverse and overlap in their characteristics.

**Conclusion:** While clustering can segment employees into two broad groups, the low silhouette scores suggest that clustering is not very meaningful for this dataset.

# -------- 6) Step 6 Answering questions  ----------

### Q1 – Why do people quit?
People are more likely to quit when they are **underpaid, overworked, or early in their career**. Long commutes and overtime further increase attrition risk.  

### Q2 – Which positions/departments are most at risk?
Lower-level roles and sales-related positions are more prone to attrition, while senior roles with higher pay are less risky.  

### Q3 – Do men and women get paid equally in all departments?
Regression results (Step 2) showed salaries are almost entirely explained by job role and level, not by gender.  
**Conclusion:** No strong evidence of systematic pay gap by gender.  

### Q4 – Does family status or distance from work affect work-life balance?
Marital status and commute distance were included, but neither stood out in the models.  
**Conclusion:** They may affect individuals, but not overall work-life balance patterns.  

### Q5 – Does education affect job satisfaction?
Education level showed little relationship with job satisfaction.  
**Conclusion:** Satisfaction depends more on role, pay, and management than education.  

### Q6 – Which ML methods did you apply and why?
- Linear Regression → for income prediction (numeric target).  
- Logistic Regression → for attrition classification (binary target).  
- Logistic Regression with weights → to handle imbalance.  
- KMeans → for unsupervised clustering and employee segmentation.  

### Q7 – How accurate are your models?
- Regression: R² = 0.938, RMSE = 1167 → very accurate.  
- Classification baseline: Accuracy 86%, Recall for quitters 0.34 → weak on attrition.  
- Classification balanced: Accuracy 76%, Recall for quitters 0.64 → better for HR.  
- Clustering: Best silhouette = 0.114 → weak clustering.  

### Q8 – How could the models be improved?
- Regression: already strong.  
- Classification: improve with oversampling, threshold tuning, or non-linear models
- Clustering: improve with feature engineering or dimensionality reduction .  

### Q9 – What challenges did you face?
- Class imbalance made attrition prediction difficult.  
- High accuracy was misleading until recall was analyzed.  
- Clustering gave poor results (weak natural groups).  
- Dataset issues (mislabelled file extension) and environment setup caused extra work.  