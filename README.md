# ROI Lens: Advanced Marketing Attribution & Budget Refinement Strategy

ROI Lens is a proprietary business intelligence and marketing attribution system built for **Nexus Consumer Brands**. By transitioning away from legacy "Last-Click" models to a **Multi-Touch Markov Chain Attribution framework**, ROI Lens uncovers hidden bot traffic, recalculates the true cost-per-acquisition (CPA) of marketing channels, and optimizes a **₹100 Crore portfolio budget** to maximize sales next quarter.

---

## 📊 Executive & Analytical Summary

### 1. The Bot Traffic Audit (Wasted Spend)
* **Key Discovery**: ROI Lens audited 566,510 raw interaction logs and identified that **₹49.23 Crore (48.33%)** of last quarter's ₹100 Crore budget was wasted on bots.
* **Pricing Model Vulnerability**:
  * **CPC Channels (Google Search, Marketplace)**: Vulnerable to automated click-spam, resulting in **~82% budget waste**.
  * **CPM Channels (Instagram, YouTube, Influencer Blog)**: Naturally safer, with only **~22% budget waste**.
* **Bots Identified**: 1,952 users were flagged as bots based on abnormally high event frequencies (60 to 140 events per user), a perfect 1.0 Click-Through Rate (every impression was clicked in the same minute), and 0 purchases.

### 2. Multi-Touch Attribution (MTA) Markov Chain Results
* **The Last-Click Fallacy**: Legacy Last-Click models heavily over-credited **Google Search** (giving it **48.40%** of conversions vs. its true Markov weight of **27.99%**).
* **Funnel Positions**:
  * **Top-of-Funnel (ToF) Primers**: **YouTube** (Markov Weight: **16.66%**) and **Influencer Blogs** (Markov Weight: **16.97%**) initiate consumer awareness. Removing them causes a **~17% drop** in overall conversion probability.
  * **Mid-Funnel Hubs**: **Instagram** (Markov Weight: **18.85%**) and **Marketplace** (Markov Weight: **19.52%**) act as bridges between interest and intent.

### 3. Legacy CPA vs. True CPA Audit (Brand B01 Example)
Stripping away bot-wasted spend reveals the true cost effectiveness of bottom-of-funnel closing channels:
* **Instagram**: Legacy CPA of ₹3,768 $\rightarrow$ **True CPA of ₹2,915** (-22.6%)
* **Google Search**: Legacy CPA of ₹211,577 $\rightarrow$ **True CPA of ₹36,839** (-82.6%)
* **Marketplace**: Legacy CPA of ₹276,299 $\rightarrow$ **True CPA of ₹44,965** (-83.7%)
* **Influencer Blog**: Legacy CPA of ₹107,631 $\rightarrow$ **True CPA of ₹82,437** (-23.4%)
* **YouTube**: Legacy CPA of ₹147,648 $\rightarrow$ **True CPA of ₹113,651** (-23.0%)

### 4. Portfolio Reallocation & Conversions Boost
Nexus authorized a fresh **₹100 Crore budget** (₹10 Crore per brand). Using calibrated diminishing returns ($Conversions = k_c \cdot Spend^{0.75}$) and SLSQP optimization, we modeled three scenarios:
* **Historical Baseline**: **5,498 conversions** recorded last quarter.
* **Scenario 1 (Bots Blocked, Unconstrained)**: **40,506.8 conversions** (+636.8% increase).
* **Scenario 2 (Bots Remain, Optimized Allocation)**: **26,408.2 conversions** (+380.3% increase).
* **Scenario 1 (Bots Blocked, Strategic Constrained)**: **31,151.0 conversions (+466.6% increase)**.

*Note: The **Strategic Constrained Portfolio** (Scenario 1 Constrained) enforces a **10% minimum budget floor** and a **50% maximum budget cap** per channel to prevent ad fatigue, protect against platform outages, and maintain full-funnel presence.*

---

## 📂 Codebase & Project Structure

The project workspace contains the following production files:

* **[roi_lens_pipeline.py](file:///c:/Users/Lenovo/Downloads/Roi%20lens/roi_lens_pipeline.py)**: A modular, command-line Python script that runs the entire data science pipeline: bot filtering, Markov Chain attribution, True CPA auditing, SLSQP portfolio optimization, and persona mapping.
* **[roi_lens_analysis.ipynb](file:///c:/Users/Lenovo/Downloads/Roi%20lens/roi_lens_analysis.ipynb)**: An interactive Jupyter Notebook illustrating the pipeline's steps, calculations, and data visualizations.
* **[roi_lens_optimized_allocations.csv](file:///c:/Users/Lenovo/Downloads/Roi%20lens/roi_lens_optimized_allocations.csv)**: Output CSV file containing the unconstrained and strategically constrained budget allocations and expected conversions for all 50 brand-channel campaigns.
* **[cmo_strategy_memo.md](file:///c:/Users/Lenovo/Downloads/Roi%20lens/cmo_strategy_memo.md)**: A formal, 9-slide executive strategy deck prepared for the Chief Marketing Officer.

---

## 🛠️ Getting Started & Usage

### 1. Prerequisites & Installation
Ensure you have Python 3.8+ installed along with the following libraries:
```bash
pip install pandas numpy scipy matplotlib seaborn
```

### 2. Running the Data Science Pipeline
To run the production pipeline script, execute:
```bash
python roi_lens_pipeline.py
```
This script will automatically:
1. Load the raw datasets (`touchpoints.csv`, `campaign_spend.csv`, `user_profiles.csv`).
2. Detect and filter out bot traffic.
3. Compute the Markov Chain multi-touch attribution weights for all 10 brands.
4. Perform the financial CPA and wasted budget audit.
5. Calibrate the power curves and solve the budget optimization problem.
6. Export the final campaign allocations to `roi_lens_optimized_allocations.csv`.
7. Print diagnostic channel and brand budget summary tables to the terminal.

### 3. Running the Jupyter Notebook
To run the interactive analysis and view plots:
```bash
jupyter notebook roi_lens_analysis.ipynb
```

---

## 🧮 Methodological Rationale

### Markov Chain Absorption Probabilities
A transition probability matrix $P$ is built from clean chronological user sequences. The transient states represent channels ($Q$ matrix), and the absorbing states represent `Purchase` or `Null` ($R$ matrix). The fundamental matrix $N$ is calculated as:
$$N = (I - Q)^{-1}$$
The absorption probability matrix $B$ (representing the probability of reaching Purchase or Null from each state) is:
$$B = N \cdot R$$
The **Removal Effect** of channel $c$ is computed by setting all transitions from $c$ to go to `Null` with probability 1.0, and measuring the resulting drop in conversion probability. The channel attribution weight is the normalized removal effect.

### Convex Budget Optimization Scaling
To prevent numerical gradient stagnation in the SLSQP solver (where gradients like $\frac{\partial f}{\partial x_i} = -0.75 \cdot k_i \cdot x_i^{-0.25}$ become tiny at Rupees scale $x_i = 10^8$), the optimizer variables are scaled to units of **₹1 Crore** ($10^7$ INR). The scaled coefficient becomes:
$$K_i = k_i \cdot (10^7)^{0.75}$$
This shifts the optimization variables to a healthy $[0.0, 10.0]$ range, ensuring robust convergence and optimal budget allocation.
