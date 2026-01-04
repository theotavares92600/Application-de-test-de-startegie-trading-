import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import datetime



# 1. Widget l'actif 
tickers = {"Engie (ENGI.PA)": "ENGI.PA","Apple (AAPL)": "AAPL","Euro/Dollar (EURUSD=X)": "EURUSD=X","Gold (GC=F)": "GC=F","Bitcoin (BTC-USD)": "BTC-USD"
}

selected_asset_name = st.selectbox("Choisir un actif à analyser :", list(tickers.keys()))
selected_ticker = tickers[selected_asset_name]

# 2. Récupération des données (Data Retrieval) 
# On utilise une fonction avec @st.cache_data pour éviter de re-télécharger à chaque clic
@st.cache_data
def load_data(ticker):
    # On change "2y" en "10y" pour avoir plus d'historique
    data = yf.download(ticker, period="10y", interval="1d")
    return data

data_load_state = st.text('Chargement des données...')
df = load_data(selected_ticker)
# --- CORRECTION DU BUG YFINANCE ---
# Si les colonnes ont deux niveaux (ex: Prix et Ticker), on aplatit pour n'avoir que le Prix
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
# ----------------------------------
data_load_state.text('Chargement des données... Terminé !')

# 3. Affichage des données brutes (Vérification)
if not df.empty:
    st.subheader(f"Données pour {selected_asset_name}")
    
    # Affichage de la valeur actuelle (Current Value) 
    last_price = df['Close'].iloc[-1]
    # On gère le formatage si c'est une série ou un scalaire (parfois yfinance renvoie des formats complexes)
    if isinstance(last_price, pd.Series):
        last_price = last_price.item()
        
    st.metric(label="Prix Actuel", value=f"{last_price:.2f}")
    
    # Afficher les dernières lignes du tableau
    st.dataframe(df.tail())
else:
    st.error("Erreur lors de la récupération des données.")
    
# --- 3. ANALYSE ET STRATÉGIES ---
st.write("---")
st.header("Analyse de Stratégies")

# Paramètres interactifs
col1, col2, col3 = st.columns(3)
with col1:
    short_window = st.slider("Moyenne Mobile Courte", 5, 50, 20)
with col2:
    long_window = st.slider("Moyenne Mobile Longue", 50, 200, 50)
with col3:
    rsi_window = st.slider("Période RSI", 5, 30, 14)

# Calcul des indicateurs (Moteur de calcul)
df = df.sort_index()
df['Returns'] = df['Close'].pct_change()

# Indicateurs Techniques
df['SMA'] = df['Close'].rolling(window=short_window).mean()
df['LMA'] = df['Close'].rolling(window=long_window).mean()

delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# Nettoyage des NaN
df_clean = df.dropna().copy()
initial_capital = 100

# --- CALCUL DES 3 STRATÉGIES ---
# 1. Buy & Hold
df_clean['BuyHold_Strategy'] = initial_capital * (1 + df_clean['Returns']).cumprod()

# 2. Moyennes Mobiles
df_clean['Signal_MA'] = 0
df_clean.loc[df_clean['SMA'] > df_clean['LMA'], 'Signal_MA'] = 1
df_clean['Ret_MA'] = df_clean['Signal_MA'].shift(1) * df_clean['Returns']
df_clean['MA_Strategy'] = initial_capital * (1 + df_clean['Ret_MA']).cumprod()

# 3. RSI
df_clean['Signal_RSI'] = 0
df_clean.loc[df_clean['RSI'] < 30, 'Signal_RSI'] = 1
df_clean.loc[df_clean['RSI'] > 70, 'Signal_RSI'] = 0
df_clean['Signal_RSI'] = df_clean['Signal_RSI'].replace(to_replace=0, method='ffill')
df_clean['Ret_RSI'] = df_clean['Signal_RSI'].shift(1) * df_clean['Returns']
df_clean['RSI_Strategy'] = initial_capital * (1 + df_clean['Ret_RSI']).cumprod()

# --- SÉLECTEUR DE STRATÉGIES (NOUVEAU) ---
st.subheader("Comparaison Dynamique")

# Dictionnaire de configuration : Nom Affiché -> {Colonne Valeur, Colonne Rendement}
strategies_config = {
    "Buy & Hold": {"val_col": "BuyHold_Strategy", "ret_col": "Returns"},
    "Moyenne Mobile": {"val_col": "MA_Strategy", "ret_col": "Ret_MA"},
    "RSI (Oscillateur)": {"val_col": "RSI_Strategy", "ret_col": "Ret_RSI"}
}

# Le Widget Multiselect
selected_strats = st.multiselect(
    "Choisissez les stratégies à afficher :",
    options=list(strategies_config.keys()),
    default=list(strategies_config.keys()) # Tout sélectionné par défaut
)

if not selected_strats:
    st.warning("Veuillez sélectionner au moins une stratégie.")
else:
    # 1. Filtrage pour le Graphique
    cols_to_plot = [strategies_config[name]["val_col"] for name in selected_strats]
    st.line_chart(df_clean[cols_to_plot])

    # 2. Filtrage pour le Tableau de Performance
    metrics_list = []
    
    for name in selected_strats:
        config = strategies_config[name]
        series_val = df_clean[config["val_col"]]
        series_ret = df_clean[config["ret_col"]]
        
        # Calculs
        total_ret = (series_val.iloc[-1] / initial_capital) - 1
        
        roll_max = series_val.cummax()
        drawdown = (series_val - roll_max) / roll_max
        max_dd = drawdown.min()
        
        sharpe = (series_ret.mean() / series_ret.std()) * (252**0.5)
        
        # Ajout à la liste
        metrics_list.append({
            "Stratégie": name,
            "Rendement Total": f"{total_ret:.2%}",
            "Max Drawdown": f"{max_dd:.2%}",
            "Ratio de Sharpe": f"{sharpe:.2f}"
        })
    
    st.table(pd.DataFrame(metrics_list))


from sklearn.linear_model import LinearRegression
import datetime

import numpy as np 

# --- BONUS : PRÉDICTION MACHINE LEARNING AVEC INTERVALLE ---
st.write("---")
st.header("🔮 Bonus : Prédiction & Intervalle de Confiance")

# Préparation des données
df_ml = df.reset_index()
df_ml['Date_Ordinal'] = df_ml['Date'].map(datetime.datetime.toordinal)

# On entraîne sur les 180 derniers jours
df_recent = df_ml.tail(180)
X = df_recent[['Date_Ordinal']]
y = df_recent['Close']

# Création et entraînement
model = LinearRegression()
model.fit(X, y)

# --- CALCUL DE LA MARGE D'ERREUR (CONFIANCE) ---
# On regarde si le modèle se trompait beaucoup sur les données passées
preds_train = model.predict(X)
residuals = y - preds_train
std_error = residuals.std() # Écart-type des erreurs

# Prédiction sur 30 jours
last_date = df_ml['Date'].iloc[-1]
future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, 31)]
future_dates_ordinal = [[d.toordinal()] for d in future_dates]

future_prices = model.predict(future_dates_ordinal)

# Construction de l'intervalle de confiance (95% = +/- 2 écarts-types)
lower_bound = future_prices - (2 * std_error)
upper_bound = future_prices + (2 * std_error)

# DataFrame Futur
df_future = pd.DataFrame({
    'Date': future_dates,
    'Scénario Bas': lower_bound,
    'Prédiction': future_prices,
    'Scénario Haut': upper_bound
})
df_future.set_index('Date', inplace=True)

st.subheader("Projection à 30 jours avec tunnel de confiance")

# --- GRAPHIQUE AMÉLIORÉ ---
# On affiche l'historique récent + les 3 courbes futures
df_history = df.tail(180)[['Close']].rename(columns={'Close': 'Historique'})
# On combine tout
df_final_chart = pd.concat([df_history, df_future], axis=1)

# Astuce visuelle : st.line_chart assigne des couleurs auto. 
# On affiche les 4 colonnes : Historique, Bas, Prédiction, Haut
st.line_chart(df_final_chart)

# Affichage des métriques
pred_val = future_prices[-1]
curr_val = df['Close'].iloc[-1]
var = (pred_val - curr_val) / curr_val

col1, col2, col3 = st.columns(3)
col1.metric("Prix Actuel", f"{curr_val:.2f} €")
col2.metric("Objectif (30j)", f"{pred_val:.2f} €", f"{var:.2%}")
col3.metric("Intervalle de confiance", f"{lower_bound[-1]:.0f} - {upper_bound[-1]:.0f} €")

st.caption("Le 'tunnel' (Scénario Haut/Bas) représente la zone où le prix a 95% de chances de se trouver si la tendance et la volatilité actuelles continuent.")