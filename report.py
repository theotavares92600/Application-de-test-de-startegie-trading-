import yfinance as yf
import pandas as pd
import datetime
import os

# --- CONFIGURATION ---
ticker = "AAPL"  # L'actif à surveiller (Apple par exemple)
file_path = "rapport_quotidien.txt" # Le nom du fichier où on stocke les rapports

# 1. Récupération des données (1 an d'historique)
# Le 'progress=False' évite de polluer les logs quand c'est automatisé
data = yf.download(ticker, period="1y", interval="1d", progress=False)

# Correction du bug MultiIndex (comme dans ton projet principal)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# 2. Calculs des Métriques (Demandé par le sujet)
last_close = data['Close'].iloc[-1]
open_price = data['Open'].iloc[-1]
returns = data['Close'].pct_change()
volatility = returns.std() * (252**0.5) # Volatilité annualisée

# Max Drawdown
roll_max = data['Close'].cummax()
drawdown = (data['Close'] - roll_max) / roll_max
max_dd = drawdown.min()

# 3. Préparation du texte
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

texte_rapport = f"""
==========================================
RAPPORT AUTOMATIQUE DU {now}
Actif surveillé : {ticker}
------------------------------------------
Prix Ouverture  : {open_price:.2f} $
Prix Clôture    : {last_close:.2f} $
Volatilité (An) : {volatility:.2%}
Max Drawdown    : {max_dd:.2%}
==========================================
"""

# 4. Écriture dans le fichier (Mode 'a' pour Append/Ajouter)
# Cela écrira à la suite sans effacer les anciens rapports
with open(file_path, "a", encoding="utf-8") as f:
    f.write(texte_rapport)

print("Rapport généré avec succès !")