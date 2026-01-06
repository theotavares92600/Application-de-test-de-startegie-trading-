import yfinance as yf
import pandas as pd
import datetime
import os

# --- CONFIGURATION ---
# Liste des actifs à inclure dans le rapport
TICKERS = ["AAPL", "BTC-USD", "GC=F", "EURUSD=X"]

# CHEMIN ABSOLU (Très important pour Cron)
# Remplacez '/home/ubuntu/...' par le résultat de la commande `pwd`
PROJECT_DIR = "/home/geheres/projet_finance" 
REPORT_FILE = os.path.join(PROJECT_DIR, "rapport_quotidien.txt")

def calculate_metrics(df):
    """Calcule Volatilité et Max Drawdown sur les données fournies"""
    if df.empty: return 0.0, 0.0
    
    # Rendements pour la volatilité
    returns = df.pct_change().dropna()
    # Volatilité annualisée (basée sur les 30 derniers jours)
    volatility = returns.std() * (252**0.5)
    
    # Max Drawdown
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_dd = drawdown.min()
    
    return volatility, max_dd

def generate_report():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # En-tête du rapport
    report_lines = [
        f"==========================================",
        f"RAPPORT AUTOMATIQUE - {now}",
        f"==========================================\n"
    ]
    
    print("Récupération des données...")
    
    for ticker in TICKERS:
        try:
            # On prend 30 jours pour calculer la volatilité récente
            data = yf.download(ticker, period="1mo", interval="1d", progress=False)
            
            # Gestion multi-index (Fix yfinance)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
                
            if not data.empty:
                last_price = data['Close'].iloc[-1]
                open_price = data['Open'].iloc[-1]
                vol, dd = calculate_metrics(data['Close'])
                
                # Ajout des métriques demandées par le sujet
                report_lines.append(f"Actif: {ticker}")
                report_lines.append(f"  - Prix Fermeture : {last_price:.2f}")
                report_lines.append(f"  - Prix Ouverture : {open_price:.2f}")
                report_lines.append(f"  - Volatilité (30j): {vol:.2%}")
                report_lines.append(f"  - Max Drawdown   : {dd:.2%}")
                report_lines.append("-" * 30)
            else:
                report_lines.append(f"Actif: {ticker} - Pas de données.")
                
        except Exception as e:
            report_lines.append(f"Actif: {ticker} - ERREUR: {str(e)}")

    report_lines.append("\nFin du rapport.\n")
    
    # Écriture dans le fichier (Mode 'a' pour append/ajouter à la suite)
    full_content = "\n".join(report_lines)
    
    with open(REPORT_FILE, "a", encoding="utf-8") as f:
        f.write(full_content)
        
    print(f"Succès ! Rapport ajouté à : {REPORT_FILE}")

if __name__ == "__main__":
    generate_report()
