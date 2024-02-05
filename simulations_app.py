#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import random
import math
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from scipy.stats import norm


# Define the generate_normal_random function
def generate_normal_random():
    u1, u2 = random.random(), random.random()
    z1 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    return z1

# Normal Distribution Simulation Function
def normal_distribution_simulation(n_samples=10):
    samples = [generate_normal_random() for _ in range(n_samples)]
    return samples


# In[2]:


# Brownian Motion Simulation Function
def brownian_motion_simulation(n_steps=1000, dt=0.01, initial_value=0, drift=0.1):
    trajectory = [initial_value]
    for _ in range(n_steps):
        dW = generate_normal_random() * math.sqrt(dt)
        new_value = trajectory[-1] + drift * dt + dW
        trajectory.append(new_value)
    return trajectory


# In[3]:


# Monte Carlo Simulation for Black-Scholes Model Function
def monte_carlo_black_scholes(S0=100, K=105, r=0.05, T=1.0, sigma=0.2, n_simulations=10000, n_steps=252):
    option_prices = []
    for _ in range(n_simulations):
        price_path = [S0]
        for _ in range(n_steps):
            dW = generate_normal_random() * math.sqrt(1/n_steps)
            price = price_path[-1] * math.exp((r - 0.5 * sigma**2) * (1/n_steps) + sigma * dW)
            price_path.append(price)
        option_payoff = max(0, price_path[-1] - K)
        option_price = option_payoff * math.exp(-r * T)
        option_prices.append(option_price)
    option_price_estimate = sum(option_prices) / n_simulations
    return option_price_estimate


# In[4]:


import streamlit as st

# Utilisez st.sidebar pour créer la barre de navigation latérale
option = st.sidebar.selectbox(
    'Sélectionnez une simulation:',
    ('Introduction', 'Distribution Normale', 'Mouvement Brownien', 'Simulation Monte Carlo', 'Cas Réel d\'Évaluation d\'Option')
)


# In[5]:


# Dans la colonne principale, affichez le contenu en fonction de l'option sélectionnée
if option == 'Introduction':
    st.title('Simulation Financière avec Streamlit')
    st.markdown("Bienvenue dans notre application de simulation financière basée sur Streamlit ! Cette application vous permet d'explorer et de comprendre différentes simulations liées à la finance et aux mathématiques financières. Vous pouvez choisir parmi trois options principales :")
    st.markdown("**Distribution Normale**")    
    st.markdown("**Mouvement Brownien**")   
    st.markdown("**Simulation Monte Carlo**")
    st.markdown("Ensuite, nous allons explorer un cas réel d'évaluation d'option en utilisant le modèle Black-Scholes et des données de marché réelles :")    
    st.markdown("**Cas Réel d\'Évaluation d\'Option**")    

    st.markdown("Sélectionnez l'option qui vous intéresse dans la barre de navigation latérale, explorez la théorie et exécutez des simulations pratiques pour approfondir votre compréhension. Profitez de cette application pour enrichir vos connaissances en finance et en modélisation financière.")    
                


elif option == 'Distribution Normale':
    st.title('Simulation de la Distribution Normale')
    st.header('Partie Théorique')
    
    # Définition de la Distribution Normale
    def definition_distribution_normale():
        definition = "**La distribution normale**, également connue sous le nom de distribution gaussienne, est une distribution de probabilité continue caractérisée par une courbe en forme de cloche symétrique. Elle est définie par deux paramètres : la moyenne (µ) et l'écart-type (σ). La distribution est centrée autour de la moyenne et sa forme est déterminée par l'écart-type."
        return definition

    # Objectifs de l'Utilisation de la Distribution Normale
    def objectifs_distribution_normale():
        objectifs = [
            "**Modélisation des données réelles.**",
            "**Utilisation en statistiques inférentielles** pour la construction d'intervalles de confiance et les tests d'hypothèses.",
            "**Analyse des erreurs et résidus** dans divers domaines scientifiques.",
            "**Prévision et simulation** dans la finance et d'autres domaines.",
            "**Évaluation de la performance** des processus.",
            "**Utilisation cruciale en théorie de la probabilité** pour décrire des distributions continues."
        ]
        return objectifs

    # Expression Mathématique de la Distribution Normale
    def expression_mathematique_distribution_normale():
        expression = "**La fonction de densité de probabilité (PDF) de la distribution normale est donnée par :**\n\n"
        expression += "f(x) = (1 / (σ√(2π))) * e^(-((x - µ)² / (2σ²)))"
        expression += "\n\n où :"
        expression += "\n - **f(x)** est la valeur de la densité de probabilité à un point x."
        expression += "\n - **µ (mu)** est la moyenne (espérance) de la distribution."
        expression += "\n - **σ (sigma)** est l'écart-type de la distribution."
        expression += "\n - **π (pi)** est la constante mathématique Pi, environ égale à 3.14159."
        expression += "\n\n La distribution est centrée autour de sa moyenne µ et sa forme est déterminée par l'écart-type σ."
        return expression

    # Exemples d'appel de fonctions
    definition = definition_distribution_normale()
    objectifs = objectifs_distribution_normale()
    expression_math = expression_mathematique_distribution_normale()

    # Affichage des résultats
    st.markdown("**Définition de la Distribution Normale :**")
    st.markdown(definition)
    st.markdown("**Objectifs de l'Utilisation de la Distribution Normale :**")
    for i, objectif in enumerate(objectifs, start=1):
        st.markdown(f"{i}. {objectif}")
    st.markdown("**Expression Mathématique de la Distribution Normale :**")
    st.markdown(expression_math)
    
    
    
    st.header('Partie Pratique')
    n_samples = st.slider('Nombre d\'échantillons', 10, 1000, 100)
    if st.button('Exécuter la simulation'):
        samples = normal_distribution_simulation(n_samples)
        st.write(samples)

        
        
elif option == 'Mouvement Brownien':
    st.title('Simulation du Mouvement Brownien')
    st.header('Partie Théorique')
    
    # Définition du Mouvement Brownien
    def definition_mouvement_brownien():
        definition = "**Le Mouvement Brownien** est un processus stochastique continu qui modélise le comportement aléatoire d'une particule en mouvement. Il a été nommé d'après le botaniste Robert Brown, qui a observé le mouvement erratique des particules de pollen dans un liquide. Le Mouvement Brownien est caractérisé par des déplacements aléatoires et imprévisibles de la particule au fil du temps."
        return definition

    # Objectifs de l'Utilisation du Mouvement Brownien
    def objectifs_mouvement_brownien():
        objectifs = [
            "**Modélisation du comportement aléatoire.**",
            "**Application en physique** pour décrire le mouvement thermique des particules.",
            "**Utilisation en finance** pour modéliser les fluctuations des prix des actifs financiers.",
            "**Simulation de phénomènes aléatoires** dans divers domaines scientifiques et d'ingénierie.",
            "**Analyse de la diffusion** de substances dans des milieux aléatoires."
        ]
        return objectifs

    # Expression Mathématique du Mouvement Brownien
    def expression_mathematique_mouvement_brownien():
        expression = "**Le Mouvement Brownien** est mathématiquement décrit comme un processus stochastique continu qui satisfait l'équation différentielle stochastique suivante :"
        expression += "\n\n"
        expression += "dX(t) = μ dt + σ dW(t)"
        expression += "\n\n où :"
        expression += "\n - **X(t)** est la position de la particule au temps t."
        expression += "\n - **μ (mu)** est la moyenne du mouvement, représentant la tendance du mouvement."
        expression += "\n - **σ (sigma)** est l'écart-type, mesurant la volatilité du mouvement."
        expression += "\n - **dW(t)** est une variable aléatoire qui suit une distribution normale, représentant le mouvement brownien."
        expression += "\n - **dt** est un élément de temps infinitésimal."
        expression += "\n\n"
        expression += "Cette équation indique que la position de la particule évolue de manière aléatoire en fonction de la tendance μ et de la volatilité σ, tout en étant influencée par un mouvement brownien aléatoire dW(t)."
        return expression

    # Exemples d'appel de fonctions
    definition_brownien = definition_mouvement_brownien()
    objectifs_brownien = objectifs_mouvement_brownien()
    expression_math_brownien = expression_mathematique_mouvement_brownien()

    # Affichage des résultats
    st.markdown("**Définition du Mouvement Brownien :**")
    st.markdown(definition_brownien)
    st.markdown("**Objectifs de l'Utilisation du Mouvement Brownien :**")
    for i, objectif in enumerate(objectifs_brownien, start=1):
        st.markdown(f"{i}. {objectif}")
    st.markdown("**Expression Mathématique du Mouvement Brownien :**")
    st.markdown(expression_math_brownien)    
    
      

    st.header('Partie Pratique')
    n_steps = st.slider('Nombre de pas', 100, 10000, 1000)
    dt = st.number_input('Intervalle de temps Delta', 0.01, 1.0, 0.01)
    if st.button('Exécuter la simulation'):
        trajectory = brownian_motion_simulation(n_steps=n_steps, dt=dt)
        plt.plot(trajectory)
        plt.xlabel('Temps')
        plt.ylabel('Valeur')
        st.pyplot(plt)

        
elif option == 'Simulation Monte Carlo':
    st.title('Simulation de Monte Carlo')
    st.header('Partie Théorique')
    
    # Définition de la Simulation Monte Carlo
    def definition_simulation_monte_carlo():
        definition = "**La Simulation Monte Carlo** est une technique de modélisation et d'analyse statistique qui repose sur la génération de données aléatoires pour résoudre des problèmes complexes. Elle doit son nom au célèbre casino de Monte Carlo à Monaco, en référence au caractère aléatoire des jeux de hasard. La Simulation Monte Carlo est utilisée pour estimer des résultats probables dans des situations complexes en effectuant un grand nombre de simulations aléatoires."
        return definition

    # Objectifs de l'Utilisation de la Simulation Monte Carlo
    def objectifs_simulation_monte_carlo():
        objectifs = [
            "**Modélisation de phénomènes complexes.**",
            "**Évaluation de risques et d'incertitudes** dans les domaines financier, scientifique et industriel.",
            "**Optimisation de processus** en testant différentes combinaisons de paramètres.",
            "**Prévision de résultats probables** dans des scénarios complexes.",
            "**Analyse de sensibilité** pour comprendre l'impact des variations de paramètres sur les résultats.",
            "**Résolution de problèmes mathématiques** difficiles ou inaccessibles par d'autres méthodes."
        ]
        return objectifs

    # Exemples d'appel de fonctions
    definition_monte_carlo = definition_simulation_monte_carlo()
    objectifs_monte_carlo = objectifs_simulation_monte_carlo()

    # Affichage des résultats
    st.markdown("**Définition de la Simulation Monte Carlo :**")
    st.markdown(definition_monte_carlo)
    st.markdown("**Objectifs de l'Utilisation de la Simulation Monte Carlo :**")
    for i, objectif in enumerate(objectifs_monte_carlo, start=1):
        st.markdown(f"{i}. {objectif}")    
    
    

    st.header('Partie Pratique')
    S0 = st.number_input('Prix initial de l\'actif', value=100)
    K = st.number_input('Prix d\'exercice', value=105)
    r = st.number_input('Taux sans risque', value=0.05)
    T = st.number_input('Durée jusqu\'à l\'échéance', value=1.0)
    sigma = st.number_input('Volatilité', value=0.2)
    n_simulations = st.slider('Nombre de simulations', 100, 10000, 1000)
    if st.button('Exécuter la simulation'):
        price_estimate = monte_carlo_black_scholes(S0=S0, K=K, r=r, T=T, sigma=sigma, n_simulations=n_simulations)
        st.write('Prix de l\'option estimé :', price_estimate)

                
                
elif option == 'Cas Réel d\'Évaluation d\'Option':
        
    # Section pour obtenir des données de marché en temps réel ou historiques
    st.header("1. Obtenir des données de marché")
    
    # Sélection du symbole de l'actif sous-jacent
    ticker_symbol = st.text_input("Symbole de l'actif sous-jacent (par exemple: AAPL, AMZN, MSFT):")

    # Vérifier si le symbole a été saisi
    if ticker_symbol:
        # Sélection de la période de données historiques
        period = st.selectbox("Période de données historiques:", ["1d", "1mo", "3mo", "6mo", "1y", "2y", "5y"])
    
        # Récupérer les données historiques
        data = yf.download(ticker_symbol, period=period)
    
        # Afficher les premières lignes des données
        st.write("Les premières lignes des données:")
        st.write(data.head())
    
        # Créer un graphique des prix de clôture
        st.write("Graphique des prix de clôture:")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data.index, data['Close'], label='Prix de clôture', color='blue')
        ax.set_xlabel('Date')
        ax.set_ylabel('Prix de clôture')
        ax.set_title(f'Prix de clôture de {ticker_symbol} sur la période {period}')
        ax.legend()
        ax.grid(True)
    
        # Afficher le graphique dans Streamlit en passant la figure explicite
        st.pyplot(fig)

        # Section pour implémenter le modèle Black-Scholes
        st.header("2. Modèle Black-Scholes")
    
        # Récupérer le prix initial (S0) en utilisant le dernier prix de clôture
        S0 = data["Close"].iloc[-1]
    
        # Sélection du prix d'exercice (K) de l'option
        K = st.number_input("Prix d'exercice de l'option (K):", value=105)
    
        # Sélection du taux d'intérêt sans risque (r)
        r = st.number_input("Taux d'intérêt sans risque (r):", value=0.05)
    
        # Sélection de la durée jusqu'à l'expiration (T) en années
        T = st.number_input("Durée jusqu'à l'expiration (T) en années:", value=1.0)
    
        # Calcul de la volatilité (sigma) en utilisant l'écart type des prix historiques
        historical_data = data["Close"].pct_change().dropna()
        sigma = historical_data.std() * np.sqrt(252)
    
        st.write("Paramètres du modèle Black-Scholes:")
        st.write("Prix initial de l'actif sous-jacent (S0):", S0)
        st.write("Prix d'exercice de l'option (K):", K)
        st.write("Taux d'intérêt sans risque (r):", r)
        st.write("Durée jusqu'à l'expiration (T):", T)
        st.write("Volatilité (sigma):", sigma)
    
        # Fonction pour calculer le prix de l'option d'achat (call option) en utilisant le modèle Black-Scholes        
        def black_scholes(S0, K, r, T, sigma, option_type):
            d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
    
            if option_type == 'call':
                option_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            elif option_type == 'put':
                option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
            else:
                option_price = None
    
            return option_price

        # Calcul du prix de l'option d'achat européenne
        call_price = black_scholes(S0, K, r, T, sigma, option_type='call')

        call_price_text = f"Prix de l'option d'achat européenne (modèle Black-Scholes) : {call_price:.2f}"
        st.markdown(f'<div style="background-color:#D3D3D3; padding:10px; border-radius:5px;">{call_price_text}</div>', unsafe_allow_html=True)
        
        # Calcul du prix de l'option de vente européenne
        put_price = black_scholes(S0, K, r, T, sigma, option_type='put')

        # Afficher le prix de l'option de vente européenne
        put_price_text = f"Prix de l'option de vente européenne (modèle Black-Scholes) : {put_price:.2f}"
        st.markdown(f'<div style="background-color:#D3D3D3; padding:10px; border-radius:5px;">{put_price_text}</div>', unsafe_allow_html=True)

        
        # Section pour développer une stratégie de trading
        st.header("3. Stratégie de trading")

        # Sélection des seuils d'achat (buying threshold) et de vente (stop-loss)
        seuil_achat = st.number_input("Seuil d'achat (buying threshold):", min_value=0.0, max_value=1.0, value=0.95, step=0.01)
        seuil_vente = st.number_input("Seuil de vente (stop-loss):", min_value=0.0, max_value=1.0, value=0.90, step=0.01)

        # Stratégie de trading basée sur les seuils d'achat et de vente
        if call_price is not None and put_price is not None:
            if call_price < seuil_achat and put_price < seuil_achat:
                decision = "Acheter à la fois l'option d'achat et l'option de vente"
            elif call_price < seuil_achat:
                decision = "Acheter l'option d'achat"
            elif put_price < seuil_achat:
                decision = "Acheter l'option de vente"
            elif call_price < seuil_vente or put_price < seuil_vente:
                decision = "Maintenir la position"
            else:
                decision = "Vendre la position (stop-loss atteint)"
        else:
            decision = "Impossible de prendre une décision sans prix d'options"

        decision_text = f"Décision de trading: {decision}"
        st.markdown(f'<div style="background-color:#D3D3D3; padding:10px; border-radius:5px;">{decision_text}</div>', unsafe_allow_html=True)


# In[ ]:




