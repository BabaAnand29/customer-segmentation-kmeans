Customer Segmentation with K-Means : 

This is a simple machine learning project where I used K-Means clustering to segment mall customers based on their age, annual income, and spending score. The goal is to identify different types of customers, which can help businesses make better marketing decisions.

---

What this project does

- Reads a small sample of mall customer data (50 rows)
- Uses `KMeans` clustering to find similar groups of customers
- Visualizes the clusters using Matplotlib, Seaborn, and Plotly
- Shows useful patterns like which customers spend more, and how they’re distributed

---

Tech stack used

- Python
- pandas, numpy
- matplotlib, seaborn (for plots)
- scikit-learn (for KMeans)
- plotly (for interactive visualization)

---

 Folder structure
customer-segmentation-kmeans/
├── data/
│ └── mall_customers_sample.csv # The dataset
├── src/
│ └── cluster_analysis.py # Main Python code
├── requirements.txt # Libraries to install
└── README.md # You're reading it!


---
How to run the project

1. Clone or download this repo
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt


python src/cluster_analysis.py
