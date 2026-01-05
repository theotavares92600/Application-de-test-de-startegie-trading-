import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import datetime
import numpy as np

# --- CONFIGURATION DE LA PAGE (DOIT ÊTRE AU TOUT DÉBUT) ---
st.set_page_config(page_title="Projet Finance Grp", layout="wide")

# ==========================================
# PARTIE 1 : CODE DE QUANT A (Fonction)
# ==========================================
@st.cache_data
def load_data_a(ticker):
    data = yf.download(ticker, period="10y", interval="1d")
    # Correction bug multi-index yfinance
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

def page_quant_a():
    st.header("📈 Analyse Univariée (Quant A)")
    
    tickers = {
        "Engie (ENGI.PA)": "ENGI.PA", "Apple (AAPL)": "AAPL",
        "Euro/Dollar (EURUSD=X)": "EURUSD=X", "Gold (GC=F)": "GC=F",
        "Bitcoin (BTC-USD)": "BTC-USD"
    }

    selected_asset_name = st.selectbox("Choisir un actif :", list(tickers.keys()))
    selected_ticker = tickers[selected_asset_name]

    df = load_data_a(selected_ticker)

    if not df.empty:
        st.metric("Prix Actuel", f"{df['Close'].iloc[-1]:.2f}")
        
        # --- STRATÉGIES ---
        st.subheader("Paramètres Stratégies")
        col1, col2, col3 = st.columns(3)
        with col1: short_window = st.slider("Moyenne Mobile Courte", 5, 50, 20)
        with col2: long_window = st.slider("Moyenne Mobile Longue", 50, 200, 50)
        with col3: rsi_window = st.slider("Période RSI", 5, 30, 14)

        # Calculs
        df['Returns'] = df['Close'].pct_change()
        df['SMA'] = df['Close'].rolling(window=short_window).mean()
        df['LMA'] = df['Close'].rolling(window=long_window).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        df_clean = df.dropna().copy()
        initial_capital = 100

        # Stratégies
        df_clean['BuyHold'] = initial_capital * (1 + df_clean['Returns']).cumprod()
        
        df_clean['Signal_MA'] = np.where(df_clean['SMA'] > df_clean['LMA'], 1, 0)
        df_clean['Ret_MA'] = df_clean['Signal_MA'].shift(1) * df_clean['Returns']
        df_clean['MA_Strat'] = initial_capital * (1 + df_clean['Ret_MA']).cumprod()

        df_clean['Signal_RSI'] = 0
        df_clean.loc[df_clean['RSI'] < 30, 'Signal_RSI'] = 1
        df_clean.loc[df_clean['RSI'] > 70, 'Signal_RSI'] = 0
        df_clean['Signal_RSI'] = df_clean['Signal_RSI'].ffill().fillna(0)
        df_clean['Ret_RSI'] = df_clean['Signal_RSI'].shift(1) * df_clean['Returns']
        df_clean['RSI_Strat'] = initial_capital * (1 + df_clean['Ret_RSI']).cumprod()

        # Graphique
        st.line_chart(df_clean[['BuyHold', 'MA_Strat', 'RSI_Strat']])
        
        # Bonus ML (Simplifié pour l'intégration)
        st.markdown("---")
        st.subheader("🔮 Prédiction (Bonus)")
        # (Code ML abrégé pour lisibilité, le principe est le même)
        df_ml = df.reset_index().tail(180)
        df_ml['Date_Ord'] = df_ml['Date'].map(datetime.datetime.toordinal)
        X = df_ml[['Date_Ord']]
        y = df_ml['Close']
        model = LinearRegression().fit(X, y)
        future_days = [[(df_ml['Date'].iloc[-1] + datetime.timedelta(days=i)).toordinal()] for i in range(1,31)]
        preds = model.predict(future_days)
        st.success(f"Prédiction à 30 jours : {preds[-1]:.2f} (Tendance calculée)")


# ==========================================
# PARTIE 2 : VOTRE CODE QUANT B (Fonction)
# ==========================================
@st.cache_data
def load_multi_data(tickers_list):
    if not tickers_list: return pd.DataFrame()
    data = yf.download(tickers_list, period="5y", interval="1d")
    if 'Close' in data.columns: return data['Close']
    return data

def page_quant_b():
    st.header("💼 Gestion de Portefeuille (Quant B)")
    
    # 1. SÉLECTION DES ACTIFS
    available_assets = {
        "Apple": "AAPL", "Microsoft": "MSFT", "Gold": "GC=F", 
        "Bitcoin": "BTC-USD", "S&P 500": "^GSPC", "Euro/USD": "EURUSD=X"
    }
    
    col_sel1, col_sel2 = st.columns([3, 1])
    with col_sel1:
        selected_names = st.multiselect(
            "Sélectionnez au moins 3 actifs :", 
            list(available_assets.keys()), 
            default=["Apple", "Gold", "Bitcoin"]
        )
    
    if len(selected_names) < 3:
        st.warning("⚠️ Il faut au moins 3 actifs.")
        return

    tickers = [available_assets[n] for n in selected_names]
    df = load_multi_data(tickers)
    
    if df.empty:
        st.error("Erreur de chargement des données.")
        return

    # Nettoyage
    df_clean = df.dropna()

    # 2. ALLOCATION DES POIDS (SLIDERS)
    st.subheader("Allocation du Capital")
    st.caption("Définissez la part de chaque actif dans votre portefeuille.")
    
    weights = []
    cols = st.columns(len(selected_names))
    
    # On crée un slider pour chaque actif choisi
    for i, name in enumerate(selected_names):
        with cols[i]:
            # Poids par défaut = 100 divisé par le nombre d'actifs
            w = st.number_input(f"{name} (%)", min_value=0, max_value=100, value=int(100/len(selected_names)), step=5)
            weights.append(w / 100) # On convertit en décimal (ex: 0.50)

    # Vérification que la somme fait 100%
    total_weight = sum(weights)
    if not (0.99 <= total_weight <= 1.01): # Petite tolérance pour les arrondis
        st.warning(f"⚠️ Attention : La somme des poids est de {total_weight*100:.0f}%. Elle devrait être de 100%.")

    # 3. CALCULS DU PORTEFEUILLE
    # A. Rendements individuels
    returns = df_clean.pct_change().dropna()
    
    # B. Rendement du Portefeuille (Produit matriciel)
    # Formule : R_port = w1*R1 + w2*R2 ...
    
    # Attention : il faut s'assurer que l'ordre des poids correspond à l'ordre des colonnes du DataFrame
    # Le DataFrame classe souvent les colonnes par ordre alphabétique des tickers
    ticker_order = df_clean.columns.tolist()
    
    # On doit réordonner notre liste de poids pour qu'elle colle aux colonnes
    ordered_weights = []
    for ticker in ticker_order:
        # On trouve le nom associé à ce ticker
        name_found = [k for k, v in available_assets.items() if v == ticker][0]
        # On trouve l'index de ce nom dans la sélection utilisateur
        index_in_selection = selected_names.index(name_found)
        # On ajoute le poids correspondant
        ordered_weights.append(weights[index_in_selection])

    returns['Portfolio'] = returns.dot(ordered_weights)

    # C. Construction de la valeur Base 100
    # On recalcule tout en base 100 (Actifs + Portefeuille)
    df_normalized = (1 + returns).cumprod() * 100
    
    # 4. VISUALISATION
    st.write("---")
    st.subheader("Performance : Portefeuille vs Actifs")
    
    # Graphique principal
    st.line_chart(df_normalized)

    
# 5. MÉTRIQUES CLÉS & DIVERSIFICATION
    st.subheader("Analyse de Risque & Rendement")
    
    # Calculs sur le Portefeuille
    port_series = df_normalized['Portfolio']
    port_ret = returns['Portfolio']
    
    # 1. Rendement (Return)
    total_ret = (port_series.iloc[-1] / 100) - 1
    annualized_ret = port_ret.mean() * 252 # Rendement moyen annualisé
    
    # 2. Volatilité (Risque)
    volatility = port_ret.std() * (252**0.5)
    
    # 3. Ratio de Sharpe
    sharpe = (port_ret.mean() / port_ret.std()) * (252**0.5)
    
    # --- CALCUL DE L'EFFET DE DIVERSIFICATION (NOUVEAU) ---
    # On calcule quelle serait la volatilité si les actifs n'étaient pas diversifiés (moyenne pondérée)
    weighted_vol = 0
    for ticker, weight in zip(ticker_order, ordered_weights):
        # Volatilité annuelle de chaque actif individuel
        asset_vol = returns[ticker].std() * (252**0.5)
        weighted_vol += asset_vol * weight
        
    # L'effet de diversification est la réduction de risque obtenue
    diversification_effect = weighted_vol - volatility

    # Affichage sur 4 colonnes pour inclure l'effet de diversification
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    col_m1.metric("Rendement Total", f"{total_ret:.2%}", help="Gain total sur la période")
    col_m2.metric("Volatilité", f"{volatility:.2%}", help="Risque réel du portefeuille")
    col_m3.metric("Ratio de Sharpe", f"{sharpe:.2f}", help="Rendement par unité de risque")
    
    # On affiche le gain de sécurité en vert
    col_m4.metric(
        "Effet Diversification", 
        f"-{diversification_effect:.2%}", 
        help="Réduction du risque grâce à la diversification (Volatilité Moyenne - Volatilité Réelle)"
    )
    
    if diversification_effect > 0:
        st.success(f"✅ La diversification fonctionne : votre risque est réduit de {diversification_effect:.2%} par rapport à la moyenne pondérée de vos actifs.")
    else:
        st.warning("⚠️ Peu d'effet de diversification (Actifs trop corrélés).")
# 6. MATRICE DE CORRÉLATION (DIVERSIFICATION)
    st.write("---")
    st.subheader("Matrice de Corrélation")
    st.markdown("""
    Cette matrice mesure la **diversification** de votre portefeuille.
    * **Proche de 1 (Rouge)** : Les actifs bougent ensemble (Peu de diversification).
    * **Proche de 0 ou Négatif (Bleu)** : Les actifs sont décorrélés (Bonne diversification).
    """)
    
    # Calcul de la corrélation sur les rendements (pas sur les prix !)
    # On ne garde que les actifs sélectionnés (pas la colonne Portfolio)
    # Note : returns contient déjà les colonnes des tickers, mais on veut afficher les noms
    
    # Petite astuce pour renommer les colonnes du ticker vers le nom lisible
    # On inverse le dictionnaire : 'AAPL' -> 'Apple'
    inv_map = {v: k for k, v in available_assets.items()}
    
    # On filtre pour ne garder que les colonnes correspondant aux tickers choisis
    # (Parfois 'returns' peut avoir d'autres colonnes si on a mal nettoyé, par sécurité on filtre)
    cols_to_corr = [t for t in tickers if t in returns.columns]
    
    corr_matrix = returns[cols_to_corr].corr()
    
    # On renomme les colonnes/index pour l'affichage
    corr_matrix.rename(columns=inv_map, index=inv_map, inplace=True)
    
    # Affichage avec un dégradé de couleurs (Heatmap)
    st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', axis=None).format("{:.2f}"))
# ==========================================
# MENU PRINCIPAL (Le Chef d'Orchestre)
# ==========================================
def main():
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Aller vers :", ["Quant A (Single Asset)", "Quant B (Portfolio)"])

    if choice == "Quant A (Single Asset)":
        page_quant_a()
    elif choice == "Quant B (Portfolio)":
        page_quant_b()

# --- LIGNE CRUCIALE : C'EST ICI QUE LE SCRIPT DÉMARRE ---
if __name__ == "__main__":
    main()
