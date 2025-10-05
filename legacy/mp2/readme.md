# MP2 – Wine Quality Data Exploration and Visualisation

##  Project Overview
This project explores the **Wine Quality dataset** (red and white wine samples).  
The goal is to clean, merge, explore, and visualise the data to answer key questions about wine quality, alcohol content, and sugar levels.

Data source: `winequality-red.xlsx` and `winequality-white.xlsx`


---

##  Steps Completed

### 1. Load and Clean Data
- Loaded both **red** and **white** wine Excel files into Pandas DataFrames.  
- Skipped the first row (dataset title).  
- Merged both datasets into one combined table with a new column:  
  - `type = "red"` or `type = "white"`.  
- Final dataset shape: **6497 rows × 13 columns**

### 2. Basic Exploration
Used Pandas summary statistics and group comparisons.

#### Average quality by type
- **Red wine:** 5.64  
- **White wine:** 5.88  
 White wine has slightly higher quality on average (~0.24 difference).

#### Average alcohol by type
- **Red wine:** 10.42%  
- **White wine:** 10.51%  
 White wine has marginally more alcohol (almost the same).

#### Average residual sugar by type
- **Red wine:** 2.54 g/dm³  
- **White wine:** 6.39 g/dm³  
 White wines are much sweeter.

---
## Step 9 – Extra Questions (Consumers vs Distributors)

As part of the exploration, we looked beyond the basic averages to see what else the data can tell us.

### 🔹 Insights for Consumers
- **Alcohol and quality**: 
  - Wines with higher alcohol content are generally rated higher. 

  - Lowest alcohol wines (≤ 9.5%) average ~5.4 quality.  
  - Highest alcohol wines (> 11.3%) average ~6.0 quality.  

- **Sweetness and quality**: 

  - Moderate sweetness (~3–8 g/dm³ residual sugar) is most popular.  
  - Very sweet wines (> 8 g/dm³) actually score lower on average (~5.75). 

- **Acidity (pH)**: Wines with slightly higher pH (less acidic) score a little better, though the differences are small.  

**Conclusion (for consumers):** Stronger wines with moderate sweetness and balanced acidity are more likely to be rated highly.

---

### 🔹 Insights for Distributors
- **Consistency**:  
  - Red wines are more consistent in quality (std dev ~0.81).  
  - White wines vary more (std dev ~0.89). 

- **Risk/Reward**:  
  - Distributors can expect less variation in red wines, while white wines may include both very good and not-so-good bottles.  

- **Sulfur dioxide (SO₂)**:  
  - Average free SO₂ ~30, total SO₂ ~115.  
  - Some wines reach very high SO₂ levels (up to 289 / 440).  
  - This matters for shelf life and preservation.  

**Conclusion (for distributors):** Red wines offer more stable quality, while white wines are less predictable but could appeal to varied markets. Monitoring SO₂ levels is important for stock decisions.

---
## Step 10 – Splitting Wines into pH Groups

We divided the wines into groups ("bins") based on their **pH** (acidity).  
This shows us which ranges of pH are the most common (highest density).

### 5 Groups
| pH Range      | Number of Wines |
|---------------|-----------------|
| 2.72 – 2.98   | 350             |
| 2.98 – 3.24   | 3344            |
| 3.24 – 3.49   | 2465            |
| 3.49 – 3.75   | 322             |
| 3.75 – 4.01   | 16              |

## Most wines are in the range **pH 2.98 – 3.24**.

---

### 10 Groups
| pH Range      | Number of Wines |
|---------------|-----------------|
| 2.72 – 2.85   | 16              |
| 2.85 – 2.98   | 334             |
| 2.98 – 3.11   | 1233            |
| 3.11 – 3.24   | 2111            |
| 3.24 – 3.36   | 1663            |
| 3.36 – 3.49   | 802             |
| 3.49 – 3.62   | 263             |
| 3.62 – 3.75   | 59              |
| 3.75 – 3.88   | 12              |
| 3.88 – 4.01   | 4               |

## When using **10 bins**, the densest group is **pH 3.11 – 3.24**.

---

### Conclusion
Most wines fall in a **narrow acidity range around pH 3.0–3.3**.  
This suggests that winemakers usually aim for this balance of acidity, as it likely appeals to both taste and quality expectations.

---

## Step 11 – Correlation Analysis

We calculated Pearson correlations (−1 to +1) between all numeric features.

###  Correlation with Quality
- **Strongest:** Alcohol (`r = +0.444`)  
  → Higher alcohol content is linked with higher wine quality.  
- **Weakest:** pH (`r = +0.020`)  
  → pH has almost no relationship with quality.

Other notable relationships:
- Volatile acidity (`r = −0.266`) → Higher volatile acidity generally means lower quality.  
- Density (`r = −0.306`) → Denser wines tend to have lower quality.  

---



---

###  Heatmap
We plotted a correlation heatmap to visualise all relationships. It clearly shows alcohol’s positive correlation with quality, and the strong dependencies between some chemical measures.

## 🛠️ Requirements
- Python (Anaconda recommended)  
- Libraries: `pandas`, `matplotlib`, `openpyxl`

Install with:
```bash
conda install pandas matplotlib openpyxl
---