import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import datetime
import os

# --- CONFIGURATION DE LA PAGE (DOIT ÊTRE AU TOUT DÉBUT) ---
st.set_page_config(page_title="Projet Finance Grp", layout="wide")
# ... autres imports ...
from streamlit_autorefresh import st_autorefresh  # <--- AJOUT 1

# ... st.set_page_config(...) ...

# --- AUTO-REFRESH (Toutes les 5 minutes) ---
count = st_autorefresh(interval=5 * 60 * 1000, key="data_refresh") # <--- AJOUT 2
# ==========================================
# PARTIE 1 : CODE DE QUANT A (Single Asset)
# ==========================================

def page_quant_a():
    st.header("📈 Analyse d'Actif (Quant A)")

    # 1. Widget l'actif 
    tickers = {
        "Engie (ENGI.PA)": "ENGI.PA",
        "Apple (AAPL)": "AAPL",
        "Euro/Dollar (EURUSD=X)": "EURUSD=X",
        "Gold (GC=F)": "GC=F",
        "Bitcoin (BTC-USD)": "BTC-USD"
    }
    
    col_asset, col_params = st.columns([3, 1])
    with col_asset:
        selected_asset_name = st.selectbox("Choisir un actif à analyser :", list(tickers.keys()))
        selected_ticker = tickers[selected_asset_name]
    
    # 2. Récupération des données (Data Retrieval) 
    @st.cache_data(ttl=300) 
    def load_data(ticker):
        # On télécharge 10 ans d'historique
        data = yf.download(ticker, period="10y", interval="1d")
        return data
    
    data_load_state = st.text('Chargement des données...')
    df = load_data(selected_ticker)
    
    # --- CORRECTION DU BUG YFINANCE ---
    # Gestion multi-index si nécessaire (bug fréquent yfinance 2024)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # ----------------------------------
    
    data_load_state.text('') # Effacer le texte de chargement
    
    # 3. Affichage des données brutes
    if not df.empty:
        # Affichage metrics en haut
        last_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        delta = last_price - prev_price
        
        if isinstance(last_price, pd.Series): last_price = last_price.item()
        
        st.metric(
            label=f"Prix Actuel ({selected_ticker})", 
            value=f"{last_price:.2f}", 
            delta=f"{delta:.2f}"
        )
    else:
        st.error("Erreur lors de la récupération des données.")
        return 
    
    # --- SIDEBAR : PARAMÈTRES ET RAPPORT ---
    st.sidebar.write("---")
    st.sidebar.header("⚙️ Paramètres Généraux")
    
    # NOUVEAU : Taux sans risque pour le Sharpe
    rf_input = st.sidebar.slider("Taux sans risque (%)", 0.0, 10.0, 4.0, step=0.1, help="Utilisé pour le calcul du Ratio de Sharpe (ex: Taux Bons du Trésor US)")
    risk_free_rate = rf_input / 100 # Conversion en décimal
    
    st.sidebar.header("📋 Rapport Quotidien")
    report_path = "rapport_quotidien.txt"
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            report_content = f.read()
        
        st.sidebar.download_button(
            label="📥 Télécharger le Rapport",
            data=report_content,
            file_name="historique_rapports.txt",
            mime="text/plain"
        )
        with st.sidebar.expander("Voir le dernier rapport"):
            st.text(report_content)
    else:
        st.sidebar.info("Aucun rapport généré. (Lancer le script report.py via Cron)")
        
    # --- 3. ANALYSE ET STRATÉGIES ---
    st.write("---")
    st.header("Comparaison de Stratégies")
    
    # Paramètres Stratégies
    with st.expander("🛠️ Configurer les indicateurs (Moyennes Mobiles & RSI)"):
        col1, col2, col3 = st.columns(3)
        short_window = col1.slider("MA Courte", 5, 50, 20)
        long_window = col2.slider("MA Longue", 50, 200, 50)
        rsi_window = col3.slider("Période RSI", 5, 30, 14)
    
    # Calcul des indicateurs
    df = df.sort_index()
    df['Returns'] = df['Close'].pct_change()
    
    # MA
    df['SMA'] = df['Close'].rolling(window=short_window).mean()
    df['LMA'] = df['Close'].rolling(window=long_window).mean()
    
    # RSI
    delta_price = df['Close'].diff()
    gain = (delta_price.where(delta_price > 0, 0)).rolling(window=rsi_window).mean()
    loss = (-delta_price.where(delta_price < 0, 0)).rolling(window=rsi_window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df_clean = df.dropna().copy()
    initial_capital = 100
    
    # --- CALCUL DES STRATÉGIES ---
    # 1. Buy & Hold
    df_clean['BuyHold_Strategy'] = initial_capital * (1 + df_clean['Returns']).cumprod()
    
    # 2. Moyennes Mobiles (Golden Cross)
    df_clean['Signal_MA'] = 0
    df_clean.loc[df_clean['SMA'] > df_clean['LMA'], 'Signal_MA'] = 1
    df_clean['Ret_MA'] = df_clean['Signal_MA'].shift(1) * df_clean['Returns']
    df_clean['MA_Strategy'] = initial_capital * (1 + df_clean['Ret_MA']).cumprod()
    
    # 3. RSI (Contrarian)
    df_clean['Signal_RSI'] = 0
    df_clean.loc[df_clean['RSI'] < 30, 'Signal_RSI'] = 1 # Survente -> Achat
    df_clean.loc[df_clean['RSI'] > 70, 'Signal_RSI'] = 0 # Surachat -> Vente/Cash
    # Remplissage des signaux (ffill pour garder la position)
    df_clean['Signal_RSI'] = df_clean['Signal_RSI'].replace(0, np.nan).ffill().fillna(0)
    df_clean['Ret_RSI'] = df_clean['Signal_RSI'].shift(1) * df_clean['Returns']
    df_clean['RSI_Strategy'] = initial_capital * (1 + df_clean['Ret_RSI']).cumprod()
    
    # --- SÉLECTION ET AFFICHAGE ---
    strategies_config = {
        "Buy & Hold": {"val_col": "BuyHold_Strategy", "ret_col": "Returns"},
        "Moyennes Mobiles": {"val_col": "MA_Strategy", "ret_col": "Ret_MA"},
        "RSI Mean Reversion": {"val_col": "RSI_Strategy", "ret_col": "Ret_RSI"}
    }
    
    selected_strats = st.multiselect(
        "Sélectionnez les stratégies à comparer :",
        options=list(strategies_config.keys()),
        default=list(strategies_config.keys())
    )
    
    if selected_strats:
        # Graphique
        cols_to_plot = [strategies_config[name]["val_col"] for name in selected_strats]
        st.line_chart(df_clean[cols_to_plot])
        
        # Tableau de Métriques
        metrics_list = []
        for name in selected_strats:
            config = strategies_config[name]
            series_val = df_clean[config["val_col"]]
            series_ret = df_clean[config["ret_col"]]
            
            # Rendement Total
            total_ret = (series_val.iloc[-1] / initial_capital) - 1
            
            # Drawdown
            roll_max = series_val.cummax()
            drawdown = (series_val - roll_max) / roll_max
            max_dd = drawdown.min()
            
            # --- CALCUL DU RATIO DE SHARPE AMÉLIORÉ ---
            annualized_return = series_ret.mean() * 252
            annualized_vol = series_ret.std() * (252**0.5)
            
            if annualized_vol != 0:
                # Formule : (Rendement - Taux Sans Risque) / Risque
                sharpe = (annualized_return - risk_free_rate) / annualized_vol
            else:
                sharpe = 0
            # ------------------------------------------
            
            metrics_list.append({
                "Stratégie": name,
                "Rendement Total": f"{total_ret:.2%}",
                "Max Drawdown": f"{max_dd:.2%}",
                "Volatilité (An)": f"{annualized_vol:.2%}",
                "Sharpe Ratio": f"{sharpe:.2f}"
            })
        
        st.table(pd.DataFrame(metrics_list).set_index("Stratégie"))
        
        st.caption(f"*Note: Le Ratio de Sharpe est calculé avec un taux sans risque de {rf_input}% (configurable dans la barre latérale).*")
    
    # --- PRÉDICTION ML (Bonus) ---
    st.write("---")
    st.header("🤖 Prédiction (Machine Learning)")
    
    df_ml = df.reset_index()
    df_ml['Date'] = pd.to_datetime(df_ml['Date']).dt.tz_localize(None) 
    df_ml['Date_Ordinal'] = df_ml['Date'].map(datetime.datetime.toordinal)
    
    # Entrainement sur les 6 derniers mois
    df_recent = df_ml.tail(180)
    X = df_recent[['Date_Ordinal']]
    y = df_recent['Close']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Calcul de l'intervalle de confiance (basique)
    preds_train = model.predict(X)
    residuals = y - preds_train
    std_error = residuals.std()
    
    # Projection futur
    last_date = df_ml['Date'].iloc[-1]
    future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, 31)]
    future_dates_ordinal = [[d.toordinal()] for d in future_dates]
    future_prices = model.predict(future_dates_ordinal)
    
    lower_bound = future_prices - (2 * std_error)
    upper_bound = future_prices + (2 * std_error)
    
    df_future = pd.DataFrame({
        'Date': future_dates,
        'Scénario Bas (95%)': lower_bound,
        'Prédiction': future_prices,
        'Scénario Haut (95%)': upper_bound
    })
    df_future.set_index('Date', inplace=True)
    
    st.line_chart(pd.concat([df_recent.set_index('Date')[['Close']], df_future], axis=1))

    # Export CSV
    csv = df_clean.to_csv().encode('utf-8')
    st.sidebar.write("---")
    st.sidebar.download_button(
        label="📥 Télécharger Données (CSV)",
        data=csv,
        file_name=f'data_{selected_ticker}.csv',
        mime='text/csv',
    )

# ==========================================
# PARTIE 2 : CODE DE QUANT B (Portfolio)
# ==========================================

@st.cache_data
def load_multi_data(tickers_list):
    if not tickers_list: return pd.DataFrame()
    # Téléchargement
    data = yf.download(tickers_list, period="5y", interval="1d")
    # Gestion du format
    if 'Close' in data.columns:
        return data['Close']
    return data

def page_quant_b():
    st.header("💼 Gestion de Portefeuille (Quant B)")
    
    # --- SIDEBAR : PARAMÈTRES STRATÉGIE ---
    st.sidebar.header("Paramètres de Stratégie")
    
    # 1. PARAMÈTRE : Taux sans risque (Pour Sharpe)
    rf_input = st.sidebar.slider("Taux sans risque (%)", 0.0, 10.0, 4.0, step=0.1, key="rf_b")
    risk_free_rate = rf_input / 100

    # 2. PARAMÈTRE : Fréquence de Rééquilibrage 
    # "Quotidien" = On maintient les poids fixes (Constant Mix)
    # "Jamais" = On laisse courir les gains (Buy & Hold)
    rebal_freq = st.sidebar.selectbox(
        "Fréquence de Rééquilibrage",
        ["Quotidien (Poids Constants)", "Jamais (Buy & Hold)"],
        help="Quotidien: On vend les gagnants pour racheter les perdants chaque jour (Poids fixes).\nJamais: On investit au début et on ne touche plus rien (Les poids dérivent)."
    )

    # 3. PARAMÈTRE : Règle d'Allocation 
    alloc_rule = st.sidebar.radio(
        "Règle d'Allocation", 
        ["Manuelle", "Équipondéré (Equal Weight)"],
        horizontal=True
    )

    # --- SÉLECTION DES ACTIFS ---
    st.subheader("1. Sélection des Actifs")
    available_assets = {
        "Apple": "AAPL", "Microsoft": "MSFT", "Gold": "GC=F",
        "Bitcoin": "BTC-USD", "S&P 500": "^GSPC", "Euro/USD": "EURUSD=X",
        "Tesla": "TSLA", "Amazon": "AMZN", "Nvidia": "NVDA"
    }

    selected_names = st.multiselect(
        "Sélectionnez vos actifs (Min 3) :",
        list(available_assets.keys()),
        default=["Apple", "Gold", "Bitcoin"]
    )

    if len(selected_names) < 3:
        st.warning("⚠️ Veuillez sélectionner au moins 3 actifs.")
        return

    tickers = [available_assets[n] for n in selected_names]
    
    with st.spinner('Chargement des données...'):
        df = load_multi_data(tickers)

    if df.empty:
        st.error("Erreur de chargement.")
        return

    df_clean = df.dropna()
    returns = df_clean.pct_change().dropna()

    # --- ALLOCATION DES POIDS ---
    st.subheader("2. Allocation du Capital")
    
    weights = []
    if alloc_rule == "Équipondéré (Equal Weight)":
        # Calcul automatique : 100% / nombre d'actifs
        weight_val = 1.0 / len(selected_names)
        weights = [weight_val] * len(selected_names)
        st.info(f"✅ Mode Équipondéré activé : Chaque actif a un poids de {weight_val*100:.2f}%")
        
    else:
        # Mode Manuel avec Sliders
        cols = st.columns(len(selected_names))
        default_w = int(100 / len(selected_names))
        
        for i, name in enumerate(selected_names):
            with cols[i]:
                w = st.number_input(f"{name} (%)", 0, 100, default_w, step=5)
                weights.append(w / 100)

        total_weight = sum(weights)
        st.progress(min(total_weight, 1.0))
        if not (0.99 <= total_weight <= 1.01):
            st.error(f"⚠️ Total : {total_weight*100:.0f}%. Il faut 100%.")
        else:
            st.success("Allocation valide (100%)")

    # --- CALCULS DE PERFORMANCE SELON LA FRÉQUENCE  ---
    
    # On aligne les poids avec l'ordre des colonnes du DataFrame
    ticker_order = df_clean.columns.tolist()
    ordered_weights = []
    for ticker in ticker_order:
        name_found = [k for k, v in available_assets.items() if v == ticker]
        if name_found:
            idx = selected_names.index(name_found[0])
            ordered_weights.append(weights[idx])
        else:
            ordered_weights.append(0)
    
    # Logique de simulation selon le rééquilibrage
    if rebal_freq == "Quotidien (Poids Constants)":
        # Méthode 1 : Rééquilibrage continu (Produit matriciel)
        # On suppose qu'on remet les poids à jour chaque jour
        port_ret = returns.dot(ordered_weights)
        port_cum_ret = (1 + port_ret).cumprod()
        port_val = port_cum_ret * 100 # Base 100
        
    else: # "Jamais (Buy & Hold)"
        # Méthode 2 : Pas de rééquilibrage
        # On calcule la courbe de CHAQUE actif individuellement
        cum_ret_assets = (1 + returns).cumprod()
        # On multiplie par le poids initial (ex: 0.5 * Perf_Apple + 0.5 * Perf_Gold)
        # Broadcasting numpy pour multiplier chaque colonne par son poids
        weighted_assets = cum_ret_assets.multiply(ordered_weights, axis=1)
        # La valeur du portefeuille est la somme des valeurs des actifs
        port_cum_ret = weighted_assets.sum(axis=1)
        # Rebaser à 100 au début (car cumprod commence après le jour 0)
        port_val = (port_cum_ret / port_cum_ret.iloc[0]) * 100
        # On recalcule les rendements quotidiens du portefeuille pour le Sharpe
        port_ret = port_val.pct_change().dropna()

    # --- VISUALISATION ---
    # --- VISUALISATION ---
    st.write("---")
    st.subheader("Comparaison : Portefeuille vs Actifs (Base 100)")
    
    # 1. Calculer la performance normalisée des actifs individuels (Base 100)
    # On reprend les rendements bruts calculés plus haut
    assets_normalized = (1 + returns).cumprod() * 100
    
    # 2. Créer un DataFrame combiné
    # On prend les actifs individuels
    combined_df = assets_normalized.copy()
    
    # On y ajoute la colonne du Portefeuille (calculée dans l'étape précédente)
    # On la nomme en MAJUSCULES pour qu'elle se distingue dans la légende
    combined_df[">>> PORTFOLIO <<<"] = port_val
    
    # 3. Affichage du graphique multi-courbes
    st.line_chart(combined_df)
    
    st.caption("Ce graphique normalise tous les actifs à 100 au départ, permettant de comparer visuellement la performance relative de votre stratégie face aux actifs isolés.")
    # --- MÉTRIQUES ---
    st.subheader("Analyse Financière")
    
    annualized_return = port_ret.mean() * 252
    annualized_volatility = port_ret.std() * (252**0.5)
    
    if annualized_volatility != 0:
        sharpe = (annualized_return - risk_free_rate) / annualized_volatility
    else:
        sharpe = 0

    # Effet Diversification (Approximation)
    # On compare la volatilité du portefeuille vs la somme pondérée des volatilités
    weighted_vol = 0
    for ticker, weight in zip(ticker_order, ordered_weights):
        asset_vol = returns[ticker].std() * (252**0.5)
        weighted_vol += asset_vol * weight

    div_effect = weighted_vol - annualized_volatility

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rendement Annuel", f"{annualized_return:.2%}")
    c2.metric("Volatilité", f"{annualized_volatility:.2%}")
    c3.metric("Sharpe Ratio", f"{sharpe:.2f}")
    c4.metric("Diversification", f"{div_effect:.2%}", help="Réduction du risque grâce à la corrélation imparfaite")

    # --- MATRICE DE CORRÉLATION  ---
    st.write("---")
    st.subheader("Matrice de Corrélation")
    inv_map = {v: k for k, v in available_assets.items()}
    cols_to_corr = [t for t in tickers if t in returns.columns]
    corr_matrix = returns[cols_to_corr].corr()
    corr_matrix.rename(columns=inv_map, index=inv_map, inplace=True)
    st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm').format("{:.2f}"))

# ==========================================
# MAIN ROUTING
# ==========================================
def main():
    st.sidebar.title("Navigation Projet")
    choice = st.sidebar.radio("Module :", ["Quant A (Single Asset)", "Quant B (Portfolio)"])

    if choice == "Quant A (Single Asset)":
        page_quant_a()
    elif choice == "Quant B (Portfolio)":
        page_quant_b()

if __name__ == "__main__":
    main()
