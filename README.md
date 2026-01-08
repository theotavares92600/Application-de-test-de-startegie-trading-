# 📈 Financial Quantitative Dashboard

Welcome to our Finance Project!

We designed this interactive dashboard to simulate a real-world quantitative research environment. Our goal was to build a tool that helps portfolio managers visualize real-time data, test trading strategies, and monitor portfolio performance—all in one place.

This project was built from scratch using Python and Streamlit, and it's currently running live on an AWS server.

## 🎯 What's Inside?

We divided the work into two specialized modules:

### 1. Single Asset Analysis (Quant A)
* **Real-time Tracking:** Fetches live data for assets like stocks or currencies using Yahoo Finance.
* **Strategy Tester:** You can run backtests (like Moving Averages) to see how a strategy would have performed historically.
* **Visuals:** We plotted the raw price vs. the strategy performance on the same chart for easy comparison.

### 2. Portfolio Management (Quant B)
* **Multi-Asset Simulation:** Allows you to combine 3+ assets into a single portfolio.
* **Deep Dive:** Displays correlation matrices and calculates volatility to help understand diversification.
* **Customization:** You can adjust weights to see how they impact total returns.

---

## 💻 Tech Stack
* **Core:** Python 3.10+ & Streamlit
* **Data:** Yahoo Finance API (`yfinance`)
* **Analysis:** Pandas, NumPy, Scikit-learn
* **Hosting:** AWS EC2 (Ubuntu Linux)

---

## 🚀 How to Run it Locally

If you want to test the code on your own machine, follow these steps:

1.  **Clone the repo:**
    ```bash
    git clone [https://github.com/theotavares92600/Application-de-test-de-startegie-trading-.git](https://github.com/theotavares92600/Application-de-test-de-startegie-trading-.git)
    cd Application-de-test-de-startegie-trading-
    ```

2.  **Install the requirements:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Launch the app:**
    ```bash
    streamlit run app.py
    ```

---

## ☁️ Deployment & Automation

We deployed the application on an **AWS EC2 instance** to ensure it runs 24/7. We used `tmux` to keep the session active in the background.

**Live Demo:** [http://35.180.63.4:8501](http://35.180.63.4:8501)

### 🕒 Automated Daily Report (Cron Job)
To meet the project requirements, we set up an automatic task that generates a report every evening.

We added this specific line to the server's crontab (`crontab -e`):

```bash
# Generates the daily report every day at 8:00 PM (20:00)
0 20 * * * /usr/bin/python3 /home/ubuntu/Application-de-test-de-startegie-trading-/projet.py >> /home/ubuntu/cron.log 2>&1
