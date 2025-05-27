import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor, AdaBoostClassifier, AdaBoostRegressor, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB # Aggiunto import per Naive Bayes
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.svm import SVC, SVR

# Funzione di utilità da aggiungere all'inizio del file, dopo le importazioni
def calcola_metriche_classificazione(y_test, predictions, y_pred_proba):
    """Calcola tutte le metriche di classificazione per un modello"""
    accuracy = accuracy_score(y_test, predictions)
    
    # Calcolo matrice di confusione
    cm = confusion_matrix(y_test, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    # Calcolo metriche principali
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)  # Sensitivity
    specificity = tn / (tn + fp)
    f1 = f1_score(y_test, predictions)
    
    # Calcolo ROC e AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "auc": roc_auc,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "cm": cm.tolist()
    }


# Aggiungiamo una funzione per addestrare automaticamente tutti i modelli con configurazioni semplificate
def addestra_tutti_modelli(X, y, X_train, X_test, y_train, y_test, task_type, n_features):
    """Addestra tutti i modelli disponibili con configurazioni di default ottimali"""
    
    st.session_state.model_performances = {}
    
    with st.spinner("Addestramento di tutti i modelli in corso..."):
        if task_type == "classification":
            # Regressione Logistica - Configurazione standard
            X_log_train = X_train[:, :min(2, n_features)]
            X_log_test = X_test[:, :min(2, n_features)]
            model_log = LogisticRegression(random_state=42)  # Parametri di default
            model_log.fit(X_log_train, y_train)
            predictions_log = model_log.predict(X_log_test)
            y_pred_proba_log = model_log.predict_proba(X_log_test)[:, 1]
            st.session_state.model_performances["Regressione Logistica"] = calcola_metriche_classificazione(
                y_test, predictions_log, y_pred_proba_log
            )
            
            # k-NN - Configurazione standard (k = sqrt(n))
            X_knn_train = X_train[:, :min(2, n_features)]
            X_knn_test = X_test[:, :min(2, n_features)]
            k_optimal = max(1, int(np.sqrt(len(X_train))))  # Regola empirica k ≈ √n
            model_knn = KNeighborsClassifier(n_neighbors=k_optimal)  # Default: weights='uniform', metric='minkowski'
            model_knn.fit(X_knn_train, y_train)
            predictions_knn = model_knn.predict(X_knn_test)
            y_pred_proba_knn = model_knn.predict_proba(X_knn_test)[:, 1]
            st.session_state.model_performances["k-Nearest Neighbors"] = calcola_metriche_classificazione(
                y_test, predictions_knn, y_pred_proba_knn
            )
            
            # Decision Tree - Configurazione bilanciata per interpretabilità
            X_tree_train = X_train[:, :min(2, n_features)]
            X_tree_test = X_test[:, :min(2, n_features)]
            model_tree_clf = DecisionTreeClassifier(max_depth=5, random_state=42)  # Profondità moderata
            model_tree_clf.fit(X_tree_train, y_train)
            predictions_tc = model_tree_clf.predict(X_tree_test)
            y_pred_proba_tc = model_tree_clf.predict_proba(X_tree_test)[:, 1]
            st.session_state.model_performances["Decision Tree"] = calcola_metriche_classificazione(
                y_test, predictions_tc, y_pred_proba_tc
            )
            
            # Neural Network (MLP) - Configurazione standard
            X_nn_train = X_train[:, :min(2, n_features)]
            X_nn_test = X_test[:, :min(2, n_features)]
            model_nn_clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)  # Configurazione standard
            model_nn_clf.fit(X_nn_train, y_train)
            predictions_nnc = model_nn_clf.predict(X_nn_test)
            y_pred_proba_nnc = model_nn_clf.predict_proba(X_nn_test)[:, 1]
            st.session_state.model_performances["Neural Network"] = calcola_metriche_classificazione(
                y_test, predictions_nnc, y_pred_proba_nnc
            )
            
            # Random Forest - Configurazione standard
            X_rf_train = X_train[:, :min(2, n_features)]
            X_rf_test = X_test[:, :min(2, n_features)]
            model_rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)  # Default ottimale
            model_rf_clf.fit(X_rf_train, y_train)
            predictions_rfc = model_rf_clf.predict(X_rf_test)
            y_pred_proba_rfc = model_rf_clf.predict_proba(X_rf_test)[:, 1]
            st.session_state.model_performances["Random Forest"] = calcola_metriche_classificazione(
                y_test, predictions_rfc, y_pred_proba_rfc
            )
            
            # SVM - Configurazione standard con kernel RBF
            X_svm_train = X_train[:, :min(2, n_features)]
            X_svm_test = X_test[:, :min(2, n_features)]
            model_svm_clf = SVC(kernel='rbf', probability=True, random_state=42)  # RBF è il più versatile
            model_svm_clf.fit(X_svm_train, y_train)
            predictions_svm = model_svm_clf.predict(X_svm_test)
            y_pred_proba_svm = model_svm_clf.predict_proba(X_svm_test)[:, 1]
            st.session_state.model_performances["SVM"] = calcola_metriche_classificazione(
                y_test, predictions_svm, y_pred_proba_svm
            )
            
            # Bagging - Configurazione standard
            X_bag_train = X_train[:, :min(2, n_features)]
            X_bag_test = X_test[:, :min(2, n_features)]
            model_bag_clf = BaggingClassifier(n_estimators=10, random_state=42)  # Default con 10 estimatori
            model_bag_clf.fit(X_bag_train, y_train)
            predictions_bc = model_bag_clf.predict(X_bag_test)
            y_pred_proba_bc = model_bag_clf.predict_proba(X_bag_test)[:, 1]
            st.session_state.model_performances["Bagging"] = calcola_metriche_classificazione(
                y_test, predictions_bc, y_pred_proba_bc
            )
            
            # AdaBoost - Configurazione standard
            X_boost_train = X_train[:, :min(2, n_features)]
            X_boost_test = X_test[:, :min(2, n_features)]
            model_ada_clf = AdaBoostClassifier(n_estimators=50, random_state=42)  # Default ottimale
            model_ada_clf.fit(X_boost_train, y_train)
            predictions_ac = model_ada_clf.predict(X_boost_test)
            y_pred_proba_ac = model_ada_clf.predict_proba(X_boost_test)[:, 1]
            st.session_state.model_performances["AdaBoost"] = calcola_metriche_classificazione(
                y_test, predictions_ac, y_pred_proba_ac
            )
            
            # Naive Bayes - Configurazione standard
            X_nb_train = X_train[:, :min(2, n_features)]
            X_nb_test = X_test[:, :min(2, n_features)]
            model_nb = GaussianNB()  # Nessun parametro da configurare
            model_nb.fit(X_nb_train, y_train)
            predictions_nb = model_nb.predict(X_nb_test)
            y_pred_proba_nb = model_nb.predict_proba(X_nb_test)[:, 1]
            st.session_state.model_performances["Naive Bayes"] = calcola_metriche_classificazione(
                y_test, predictions_nb, y_pred_proba_nb
            )
            
        elif task_type == "regression":
            # Regressione Lineare - Configurazione standard
            X_reg_train = X_train[:, 0].reshape(-1, 1)
            X_reg_test = X_test[:, 0].reshape(-1, 1)
            model_reg = LinearRegression()  # Nessun parametro da configurare
            model_reg.fit(X_reg_train, y_train)
            predictions_reg = model_reg.predict(X_reg_test)
            mse_reg = mean_squared_error(y_test, predictions_reg)
            r2_reg = r2_score(y_test, predictions_reg)
            st.session_state.model_performances["Regressione Lineare"] = {
                "mse": mse_reg,
                "r2": r2_reg
            }
            
            # Decision Tree Regressor - Configurazione bilanciata
            X_tree_train = X_train[:, 0].reshape(-1, 1)
            X_tree_test = X_test[:, 0].reshape(-1, 1)
            model_tree_reg = DecisionTreeRegressor(max_depth=5, random_state=42)  # Profondità moderata
            model_tree_reg.fit(X_tree_train, y_train)
            predictions_tr = model_tree_reg.predict(X_tree_test)
            mse_tr = mean_squared_error(y_test, predictions_tr)
            r2_tr = r2_score(y_test, predictions_tr)
            st.session_state.model_performances["Decision Tree"] = {
                "mse": mse_tr,
                "r2": r2_tr
            }
            
            # MLP Regressor - Configurazione standard
            X_nn_train = X_train[:, 0].reshape(-1, 1)
            X_nn_test = X_test[:, 0].reshape(-1, 1)
            model_nn_reg = MLPRegressor(hidden_layer_sizes=(100,), max_iter=300, random_state=42)  # Configurazione standard
            model_nn_reg.fit(X_nn_train, y_train)
            predictions_nnr = model_nn_reg.predict(X_nn_test)
            mse_nnr = mean_squared_error(y_test, predictions_nnr)
            r2_nnr = r2_score(y_test, predictions_nnr)
            st.session_state.model_performances["Neural Network"] = {
                "mse": mse_nnr,
                "r2": r2_nnr
            }
            
            # Random Forest Regressor - Configurazione standard
            from sklearn.ensemble import RandomForestRegressor
            X_rf_train = X_train[:, 0].reshape(-1, 1)
            X_rf_test = X_test[:, 0].reshape(-1, 1)
            model_rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)  # Default ottimale
            model_rf_reg.fit(X_rf_train, y_train)
            predictions_rfr = model_rf_reg.predict(X_rf_test)
            mse_rfr = mean_squared_error(y_test, predictions_rfr)
            r2_rfr = r2_score(y_test, predictions_rfr)
            st.session_state.model_performances["Random Forest"] = {
                "mse": mse_rfr,
                "r2": r2_rfr
            }
            
            # SVR - Configurazione standard
            X_svm_train = X_train[:, 0].reshape(-1, 1)
            X_svm_test = X_test[:, 0].reshape(-1, 1)
            model_svm_reg = SVR(kernel='rbf')  # RBF è il più versatile
            model_svm_reg.fit(X_svm_train, y_train)
            predictions_svr = model_svm_reg.predict(X_svm_test)
            mse_svr = mean_squared_error(y_test, predictions_svr)
            r2_svr = r2_score(y_test, predictions_svr)
            st.session_state.model_performances["SVM"] = {
                "mse": mse_svr,
                "r2": r2_svr
            }
            
            # Bagging Regressor - Configurazione standard
            X_bag_train = X_train[:, 0].reshape(-1, 1)
            X_bag_test = X_test[:, 0].reshape(-1, 1)
            model_bag_reg = BaggingRegressor(n_estimators=10, random_state=42)  # Default con 10 estimatori
            model_bag_reg.fit(X_bag_train, y_train)
            predictions_br = model_bag_reg.predict(X_bag_test)
            mse_br = mean_squared_error(y_test, predictions_br)
            r2_br = r2_score(y_test, predictions_br)
            st.session_state.model_performances["Bagging"] = {
                "mse": mse_br,
                "r2": r2_br
            }
            
            # AdaBoost Regressor - Configurazione standard
            X_boost_train = X_train[:, 0].reshape(-1, 1)
            X_boost_test = X_test[:, 0].reshape(-1, 1)
            model_ada_reg = AdaBoostRegressor(n_estimators=50, random_state=42)  # Default ottimale
            model_ada_reg.fit(X_boost_train, y_train)
            predictions_ar = model_ada_reg.predict(X_boost_test)
            mse_ar = mean_squared_error(y_test, predictions_ar)
            r2_ar = r2_score(y_test, predictions_ar)
            st.session_state.model_performances["AdaBoost"] = {
                "mse": mse_ar,
                "r2": r2_ar
            }
    
    return len(st.session_state.model_performances)
    
# Configurazione della pagina
st.set_page_config(layout="wide", page_title="ML Models Explainer")

st.title("Spiegazione Didattica dei Modelli di Machine Learning")

st.sidebar.header("Seleziona un Modello")
model_options = [
    "Home",
    "Dataset Generator",
    "Generalized Linear Models (GLM)",
    "k-Nearest Neighbors (kNN)",
    "Decision Trees",
    "Neural Networks",
    "Ensemble Methods: Bagging",
    "Ensemble Methods: Boosting",
    "Random Forest",
    "Support Vector Machines (SVM)",
    "Naive Bayes",
    "Riepilogo Performance"
]
selected_model = st.sidebar.selectbox("Scegli un modello da esplorare:", model_options)

# Funzione per generare dati
def generate_data(task_type="classification", n_samples=100, n_features=2, n_classes=2, random_state=42):
    if task_type == "classification":
        X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_features, n_redundant=0, 
                                   n_clusters_per_class=1, n_classes=n_classes, random_state=random_state)
    elif task_type == "regression":
        X, y = make_regression(n_samples=n_samples, n_features=n_features, n_informative=n_features, 
                               noise=20, random_state=random_state)
    else:
        raise ValueError("task_type deve essere 'classification' o 'regression'")
    
    # Standardizza le features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

# --- Pagina Home ---
if selected_model == "Home":
    st.header("Benvenuto!")
    st.markdown("""
    Questa applicazione è progettata per fornire una spiegazione didattica e interattiva
    di diversi modelli di Machine Learning con un focus sulla semplicità e l'interpretabilità.

    **Procedura di utilizzo:**
    1. Vai alla sezione "Dataset Generator" e crea un dataset.
    2. Visita le sezioni dei diversi modelli per addestrarli sullo stesso dataset.
    3. Verifica la sezione "Riepilogo Performance" per confrontare i risultati.
    
    Utilizza il menu a sinistra per navigare tra le diverse sezioni.
    """)
    
    if "dataset" in st.session_state:
        st.success("Hai già generato un dataset. Puoi procedere con l'addestramento dei modelli!")
    else:
        st.info("Per iniziare, vai alla sezione 'Dataset Generator' per creare un dataset.")
    
    st.subheader("Dataset Generati")
    st.markdown("""
    I dataset utilizzati sono generati casualmente con `scikit-learn` e standardizzati.
    Per semplicità, molte configurazioni del dataset (come il numero di features) sono preimpostate
    per ottimizzare le visualizzazioni e la comprensione.
    """)

# --- Dataset Generator ---
elif selected_model == "Dataset Generator":
    st.header("Generazione del Dataset")
    st.markdown("""
    In questa sezione puoi generare un dataset che sarà utilizzato per tutti i modelli.
    Questo permetterà di confrontare le performance di diversi algoritmi sugli stessi dati.
    """)
    
    # Parametri del dataset
    task_type = st.selectbox("Scegli il tipo di task:", ["classification", "regression"], key="dataset_task_type")
    st.caption("Classificazione: predire categorie. Regressione: predire valori numerici continui.")
    
    n_samples = st.slider("Numero di campioni:", 50, 500, 150, key="dataset_samples")
    st.caption("Determina quanti esempi verranno generati nel dataset. Più campioni = modelli potenzialmente più accurati ma addestramento più lento.")
    
    n_features = st.slider("Numero di features:", 1, 5, 2, key="dataset_features")
    st.caption("Rappresenta quante variabili di input (caratteristiche) avrà ogni esempio. Più features = modello potenzialmente più complesso.")
    
    if task_type == "classification":
        n_classes = st.slider("Numero di classi:", 2, 5, 2, key="dataset_classes")
        st.caption("Solo per classificazione: indica quante categorie distinte il modello dovrà imparare a distinguere.")
    else:
        n_classes = 2  # Non rilevante per la regressione
    
    random_seed = st.slider("Seed casuale:", 1, 100, 42, key="dataset_seed")
    st.caption("Controlla la generazione dei dati casuali. Usare lo stesso valore produce sempre lo stesso dataset, permettendo esperimenti riproducibili.")
    
    if st.button("Genera Dataset"):
        X, y = generate_data(task_type=task_type, n_samples=n_samples, 
                             n_features=n_features, n_classes=n_classes, 
                             random_state=random_seed)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_seed)
        
        # Salva il dataset nella session state
        st.session_state.dataset = {
            "X": X,
            "y": y,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "task_type": task_type,
            "n_features": n_features,
            "n_classes": n_classes
        }
        
        # Mostra una preview del dataset
        df_preview = pd.DataFrame(X, columns=[f"Feature {i+1}" for i in range(n_features)])
        df_preview["Target"] = y
        st.write("Anteprima del dataset:")
        st.dataframe(df_preview.head(10))
        
        # Visualizzazione
        if n_features >= 2 and task_type == "classification":
            fig = px.scatter(df_preview, x="Feature 1", y="Feature 2", color="Target",
                            color_discrete_sequence=px.colors.qualitative.Set1,
                            title="Visualizzazione delle prime due features")
            st.plotly_chart(fig)
        elif n_features >= 1 and task_type == "regression":
            fig = px.scatter(df_preview, x="Feature 1", y="Target",
                            title="Relazione tra Feature 1 e Target")
            st.plotly_chart(fig)
        
        # Addestra automaticamente tutti i modelli
        n_modelli = addestra_tutti_modelli(X, y, X_train, X_test, y_train, y_test, task_type, n_features)
        
        st.success(f"Dataset generato con successo! Sono stati addestrati automaticamente {n_modelli} modelli. Puoi visualizzare le performance nella sezione 'Riepilogo Performance'.")

# --- Modelli ---

if selected_model == "Generalized Linear Models (GLM)":
    st.header("Generalized Linear Models (GLM)")
    st.markdown("""
    I Modelli Lineari Generalizzati (GLM) estendono la regressione lineare per gestire diversi tipi di dati di risposta.
    I due casi più comuni sono:
    - **Regressione Lineare:** Per predire un valore numerico continuo usando la funzione identità come link.
    - **Regressione Logistica:** Per predire una categoria usando la funzione logit come link.
    
    **Configurazioni utilizzate:**
    - **Regressione Lineare:** Parametri di default (nessuna regolarizzazione)
    - **Regressione Logistica:** Solver 'lbfgs' (ottimale per dataset piccoli-medi), max_iter=100
    """)

    if "dataset" not in st.session_state:
        st.warning("Devi prima generare un dataset! Vai alla sezione 'Dataset Generator'.")
    else:
        # Estrai il dataset dalla session_state
        X = st.session_state.dataset["X"]
        y = st.session_state.dataset["y"]
        X_train = st.session_state.dataset["X_train"]
        X_test = st.session_state.dataset["X_test"]
        y_train = st.session_state.dataset["y_train"]
        y_test = st.session_state.dataset["y_test"]
        task_type = st.session_state.dataset["task_type"]
        n_features = st.session_state.dataset["n_features"]

        if task_type == "regression":
            st.subheader("Regressione Lineare")
            st.markdown("""
            **Obiettivo:** Trovare la linea retta che meglio descrive la relazione tra feature (X) e target continuo (y).
            
            **Equazione:** `y = β₀ + β₁X + ε`
            - `β₀` (Intercetta): Il valore di y quando X è 0
            - `β₁` (Coefficiente): Variazione di y per un aumento unitario di X
            - `ε` (Errore): Termine di errore casuale
            """)
            
            # Utilizziamo solo la prima feature per una visualizzazione più semplice
            X_reg_train = X_train[:, 0].reshape(-1, 1)
            X_reg_test = X_test[:, 0].reshape(-1, 1)

            model_reg = LinearRegression()  # Configurazione standard
            model_reg.fit(X_reg_train, y_train)
            predictions_reg = model_reg.predict(X_reg_test)
            
            mse_reg = mean_squared_error(y_test, predictions_reg)
            r2_reg = r2_score(y_test, predictions_reg)
            
            st.write(f"**Coefficiente (β₁):** {model_reg.coef_[0]:.4f}")
            st.write(f"**Intercetta (β₀):** {model_reg.intercept_:.4f}")
            st.write(f"*Interpretazione:* Per ogni aumento di 1 unità nella feature, il target {('aumenta' if model_reg.coef_[0] > 0 else 'diminuisce')} di {abs(model_reg.coef_[0]):.4f} unità.")
            st.write(f"**Mean Squared Error (MSE):** {mse_reg:.4f}")
            st.write(f"**R² Score:** {r2_reg:.4f} (Proporzione di varianza spiegata dal modello)")

            st.markdown("#### Visualizzazione del Modello")
            fig_reg = go.Figure()
            fig_reg.add_trace(go.Scatter(x=X_reg_test[:,0], y=y_test, mode='markers', name='Dati di Test'))
            x_range_reg = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
            y_range_reg = model_reg.predict(x_range_reg.reshape(-1, 1))
            fig_reg.add_trace(go.Scatter(x=x_range_reg, y=y_range_reg, mode='lines', name='Linea di Regressione'))
            fig_reg.update_layout(title="Regressione Lineare: Dati e Modello", xaxis_title="Feature (X)", yaxis_title="Target (y)")
            st.plotly_chart(fig_reg)

            st.markdown("#### Testa il Modello con un Nuovo Dato")
            new_x_reg_val = st.number_input("Inserisci un valore per la Feature X:", value=float(X[:, 0].mean()), format="%.2f", key="new_x_reg_val")
            if st.button("Predici con Regressione Lineare", key="pred_reg_btn"):
                predicted_y = model_reg.predict(np.array([[new_x_reg_val]]))
                st.success(f"Per X = {new_x_reg_val:.2f}, il valore predetto di y è: {predicted_y[0]:.4f}")
                
            # Salva le metriche di performance
            if "model_performances" not in st.session_state:
                st.session_state.model_performances = {}
                
            st.session_state.model_performances["Regressione Lineare"] = {
                "mse": mse_reg,
                "r2": r2_reg
            }

        elif task_type == "classification":
            st.subheader("Regressione Logistica")
            st.markdown("""
            **Obiettivo:** Predire la probabilità che un'istanza appartenga a una classe usando la funzione logistica (sigmoide).
            
            **Equazione:** `P(y=1|X) = 1 / (1 + e^(-(β₀ + β₁X₁ + β₂X₂ + ...)))`
            
            **Interpretazione dei coefficienti:**
            - `β₀` (Intercetta): Log-odds quando tutte le features sono 0
            - `βᵢ` (Coefficiente): Variazione nel log-odds per un aumento unitario di Xᵢ
            - **Odds Ratio (OR):** `exp(βᵢ)` - Come cambiano le odds per un aumento unitario di Xᵢ
            """)
            
            # Utilizziamo max due features per la visualizzazione
            X_log_train = X_train[:, :min(2, n_features)]
            X_log_test = X_test[:, :min(2, n_features)]

            model_log = LogisticRegression(solver='lbfgs', random_state=42) # Parametri di default comuni
            model_log.fit(X_log_train, y_train)
            predictions_log = model_log.predict(X_log_test)
            y_pred_proba_log = model_log.predict_proba(X_log_test)[:, 1]  # Classe positiva
            
            st.write(f"**Accuratezza:** {accuracy_score(y_test, predictions_log):.4f}")
            st.write(f"**Intercetta (β₀):** {model_log.intercept_[0]:.4f} (Log-odds quando le features sono 0)")
            st.write("**Coefficienti (β) e Odds Ratios (OR):**")
            odds_ratios = np.exp(model_log.coef_[0])
            for i, coef in enumerate(model_log.coef_[0]):
                st.write(f"  - Feature {i+1} (β{i+1}): {coef:.4f} (variazione log-odds)")
                st.write(f"    - Odds Ratio (OR): {odds_ratios[i]:.4f}. Per un aumento unitario della Feature {i+1}, le odds della classe 1 cambiano di un fattore {odds_ratios[i]:.4f}.")

            if n_features >= 2:
                st.markdown("#### Visualizzazione del Modello (con Decision Boundary)")
                df_test_log = pd.DataFrame(X_log_test, columns=['Feature 1', 'Feature 2'])
                df_test_log['Target'] = y_test

                fig_log = px.scatter(df_test_log, x='Feature 1', y='Feature 2', color='Target',
                                     color_discrete_map={0: 'blue', 1: 'red'},
                                     title="Regressione Logistica: Dati di Test e Decision Boundary")
                
                x_min_log, x_max_log = X[:, 0].min() - .5, X[:, 0].max() + .5
                y_min_log, y_max_log = X[:, 1].min() - .5, X[:, 1].max() + .5
                h_log = .02
                xx_log, yy_log = np.meshgrid(np.arange(x_min_log, x_max_log, h_log), np.arange(y_min_log, y_max_log, h_log))
                Z_log = model_log.predict(np.c_[xx_log.ravel(), yy_log.ravel()])
                Z_log = Z_log.reshape(xx_log.shape)
                
                # Assicurati che colorscale per contour sia compatibile
                contour_colors = [[0, 'lightblue'], [1, 'lightcoral']] 
                fig_log.add_trace(go.Contour(x=xx_log[0], y=yy_log[:,0], z=Z_log, showscale=False, colorscale=contour_colors, opacity=0.4, hoverinfo='skip'))
                st.plotly_chart(fig_log)

            st.markdown("#### Testa il Modello con Nuovi Dati")
            new_x_values = []
            for i in range(min(2, n_features)):
                new_x_values.append(st.number_input(f"Inserisci valore per Feature {i+1}:", value=float(X[:, i].mean()), format="%.2f", key=f"new_x{i+1}_log_val"))
            
            if st.button("Predici con Regressione Logistica", key="pred_log_btn"):
                new_data_log = np.array([new_x_values])
                pred_class = model_log.predict(new_data_log)
                pred_proba = model_log.predict_proba(new_data_log)
                st.success(f"Per i valori inseriti:")
                st.write(f"  - Classe Predetta: {pred_class[0]}")
                st.write(f"  - Probabilità (Classe 0): {pred_proba[0][0]:.4f}")
                st.write(f"  - Probabilità (Classe 1): {pred_proba[0][1]:.4f}")
                
            # Salva le metriche di performance
            if "model_performances" not in st.session_state:
                st.session_state.model_performances = {}
                
            st.session_state.model_performances["Regressione Logistica"] = calcola_metriche_classificazione(
                y_test, predictions_log, y_pred_proba_log
            )
        else:
            st.error("Tipo di task non riconosciuto. Genera un nuovo dataset.")

# --- Riepilogo Performance ---
elif selected_model == "Riepilogo Performance":
    st.header("Riepilogo delle Performance dei Modelli")
    
    if "model_performances" not in st.session_state or not st.session_state.model_performances:
        st.warning("Non hai ancora addestrato nessun modello. Vai alle sezioni dei modelli e addestra almeno un modello.")
    elif "dataset" not in st.session_state:
        st.warning("Non hai ancora generato un dataset. Vai alla sezione 'Dataset Generator'.")
    else:
        st.write("Confronto delle performance su tutti i modelli addestrati sullo stesso dataset:")
        
        task_type = st.session_state.dataset["task_type"]
        
        if task_type == "classification":
            # Crea dataframe per le metriche di classificazione
            performances = {
                "Modello": [],
                "Accuratezza": [],
                "Precision": [],
                "Recall (Sensitivity)": [],
                "Specificity": [],
                "F1-Score": [],
                "AUC": []
            }
            
            for model_name, metrics in st.session_state.model_performances.items():
                if "accuracy" in metrics:
                    performances["Modello"].append(model_name)
                    performances["Accuratezza"].append(metrics["accuracy"])
                    
                    # Aggiungi le altre metriche se disponibili
                    performances["Precision"].append(metrics.get("precision", float('nan')))
                    performances["Recall (Sensitivity)"].append(metrics.get("recall", float('nan')))
                    performances["Specificity"].append(metrics.get("specificity", float('nan')))
                    performances["F1-Score"].append(metrics.get("f1", float('nan')))
                    performances["AUC"].append(metrics.get("auc", float('nan')))
            
            df_performances = pd.DataFrame(performances)
            
            if not df_performances.empty:
                # Visualizza grafico a barre per Accuratezza
                fig = px.bar(df_performances, x="Modello", y="Accuratezza", 
                            title="Accuratezza dei Modelli",
                            color="Accuratezza", 
                            color_continuous_scale=px.colors.sequential.Viridis)
                st.plotly_chart(fig)
                
                # Mostra tabella comparativa con tutte le metriche
                st.subheader("Tabella Comparativa delle Metriche")
                df_sorted = df_performances.sort_values("Accuratezza", ascending=False)
                st.dataframe(df_sorted.style.format({
                    "Accuratezza": "{:.4f}", 
                    "Precision": "{:.4f}", 
                    "Recall (Sensitivity)": "{:.4f}", 
                    "Specificity": "{:.4f}", 
                    "F1-Score": "{:.4f}", 
                    "AUC": "{:.4f}"
                }))
                
                # Miglior modello
                best_model = df_sorted.iloc[0]["Modello"]
                best_accuracy = df_sorted.iloc[0]["Accuratezza"]
                st.success(f"Il modello con la migliore accuratezza è **{best_model}** con un valore di **{best_accuracy:.4f}**.")
                
                # Spiegazione delle metriche
                st.subheader("Spiegazione delle Metriche di Classificazione")
                
                st.markdown("""
                Per comprendere le metriche di classificazione, è utile conoscere la **matrice di confusione**, che organizza le predizioni in quattro categorie:
                
                | | Classe Predetta Positiva | Classe Predetta Negativa |
                |---|---|---|
                | **Classe Reale Positiva** | True Positive (TP) | False Negative (FN) |
                | **Classe Reale Negativa** | False Positive (FP) | True Negative (TN) |
                
                Basandosi su questi valori, le metriche sono calcolate come segue:
                
                - **Accuracy (Accuratezza):** Percentuale di predizioni corrette sul totale.
                  - Formula: (TP + TN) / (TP + TN + FP + FN)
                  - Significato: Misura generale della correttezza del modello.
                
                - **Sensitivity (Recall):** Percentuale di veri positivi correttamente identificati.
                  - Formula: TP / (TP + FN)
                  - Significato: Capacità del modello di identificare tutti i casi positivi.
                
                - **Specificity:** Percentuale di veri negativi correttamente identificati.
                  - Formula: TN / (TN + FP)
                  - Significato: Capacità del modello di escludere correttamente i casi negativi.
                
                - **Precision:** Percentuale di predizioni positive che sono effettivamente corrette.
                  - Formula: TP / (TP + FP)
                  - Significato: Affidabilità delle predizioni positive del modello.
                
                - **F1-Score:** Media armonica di precision e recall.
                  - Formula: 2 * (Precision * Recall) / (Precision + Recall)
                  - Significato: Bilanciamento tra precision e recall, utile quando le classi sono sbilanciate.
                
                - **ROC (Receiver Operating Characteristic):** Curva che rappresenta la performance del modello a diverse soglie di classificazione.
                  - Grafico: Per ogni soglia, si rappresenta (Sensitivity, 1 - Specificity)
                
                - **AUC (Area Under the Curve):** Area sotto la curva ROC.
                  - Valore: Tra 0.5 (predizione casuale) e 1.0 (predizione perfetta)
                  - Significato: Misura complessiva della capacità discriminativa del modello indipendentemente dalla soglia.
                """)
                
                # Se c'è un modello selezionato, visualizza la curva ROC
                if len(df_performances) > 0:
                    st.subheader("Curve ROC")
                    # Qui inseriremo la visualizzazione delle curve ROC, se disponibili nei dati delle performance
                    # Questo richiede modifiche anche al codice di addestramento per salvare i dati necessari
                    
                    # Se almeno un modello ha i dati per ROC
                    has_roc_data = any(["fpr" in metrics and "tpr" in metrics for model_name, metrics in st.session_state.model_performances.items()])
                    
                    if has_roc_data:
                        fig_roc = go.Figure()
                        fig_roc.add_shape(
                            type='line', line=dict(dash='dash'),
                            x0=0, x1=1, y0=0, y1=1
                        )
                        
                        for model_name, metrics in st.session_state.model_performances.items():
                            if "fpr" in metrics and "tpr" in metrics and "auc" in metrics:
                                fig_roc.add_trace(go.Scatter(
                                    x=metrics["fpr"], y=metrics["tpr"],
                                    name=f"{model_name} (AUC={metrics['auc']:.4f})",
                                    mode='lines'
                                ))
                        
                        fig_roc.update_layout(
                            xaxis_title='1 - Specificity (False Positive Rate)',
                            yaxis_title='Sensitivity (True Positive Rate)',
                            yaxis=dict(scaleanchor="x", scaleratio=1),
                            xaxis=dict(constrain='domain'),
                            title='Curve ROC per i diversi modelli'
                        )
                        st.plotly_chart(fig_roc)
                        
                        # Aggiungi visualizzazione della matrice di confusione
                        st.subheader("Matrice di Confusione")
                        
                        # Selezione del modello per visualizzare la matrice
                        modelli_con_cm = [model_name for model_name, metrics in st.session_state.model_performances.items() 
                                         if "cm" in metrics]
                        
                        if modelli_con_cm:
                            modello_selezionato = st.selectbox("Seleziona un modello:", modelli_con_cm)
                            cm = np.array(st.session_state.model_performances[modello_selezionato]["cm"])
                            
                            fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negativo', 'Positivo'], 
                                       yticklabels=['Negativo', 'Positivo'], ax=ax_cm)
                            plt.ylabel('Classe Reale')
                            plt.xlabel('Classe Predetta')
                            plt.title(f'Matrice di Confusione - {modello_selezionato}')
                            st.pyplot(fig_cm)
                            
                            # Estrai metriche dal modello selezionato
                            metrics = st.session_state.model_performances[modello_selezionato]
                            tn, fp, fn, tp = cm.ravel()
                            
                            # Mostra le formule e i calcoli per quel modello specifico
                            st.markdown(f"""
                            ### Calcoli per {modello_selezionato}:
                            
                            - **True Positive (TP)**: {tp} esempi positivi correttamente classificati
                            - **True Negative (TN)**: {tn} esempi negativi correttamente classificati
                            - **False Positive (FP)**: {fp} esempi negativi erroneamente classificati come positivi
                            - **False Negative (FN)**: {fn} esempi positivi erroneamente classificati come negativi
                            
                            - **Accuracy**: (TP + TN) / (TP + TN + FP + FN) = ({tp} + {tn}) / ({tp} + {tn} + {fp} + {fn}) = {metrics['accuracy']:.4f}
                            - **Precision**: TP / (TP + FP) = {tp} / ({tp} + {fp}) = {metrics['precision']:.4f}
                            - **Recall (Sensitivity)**: TP / (TP + FN) = {tp} / ({tp} + {fn}) = {metrics['recall']:.4f}
                            - **Specificity**: TN / (TN + FP) = {tn} / ({tn} + {fp}) = {metrics['specificity']:.4f}
                            - **F1-Score**: 2 * (Precision * Recall) / (Precision + Recall) = 2 * ({metrics['precision']:.4f} * {metrics['recall']:.4f}) / ({metrics['precision']:.4f} + {metrics['recall']:.4f}) = {metrics['f1']:.4f}
                            """)
                    else:
                        st.info("Dati insufficienti per visualizzare le curve ROC. Riaddestra i modelli per generare questi dati.")
            else:
                st.info("Non ci sono ancora modelli di classificazione addestrati.")
            
        else:  # regression
            # Crea dataframe per MSE e R²
            performances = {
                "Modello": [],
                "MSE": [],
                "R²": []
            }
            
            for model_name, metrics in st.session_state.model_performances.items():
                if "mse" in metrics and "r2" in metrics:
                    performances["Modello"].append(model_name)
                    performances["MSE"].append(metrics["mse"])
                    performances["R²"].append(metrics["r2"])
            
            df_performances = pd.DataFrame(performances)
            
            if not df_performances.empty:
                # Visualizza grafici
                fig1 = px.bar(df_performances, x="Modello", y="MSE", 
                            title="Mean Squared Error (MSE) - Valori più bassi sono migliori",
                            color="MSE", 
                            color_continuous_scale=px.colors.sequential.Viridis_r)
                st.plotly_chart(fig1)
                
                fig2 = px.bar(df_performances, x="Modello", y="R²", 
                            title="R² Score - Valori più alti sono migliori",
                            color="R²", 
                            color_continuous_scale=px.colors.sequential.Viridis)
                st.plotly_chart(fig2)
                
                # Mostra tabella ordinata
                st.subheader("Tabella Comparativa")
                df_sorted_mse = df_performances.sort_values("MSE")
                df_sorted_r2 = df_performances.sort_values("R²", ascending=False)
                
                st.write("Ordinato per MSE (crescente):")
                st.table(df_sorted_mse)
                
                st.write("Ordinato per R² (decrescente):")
                st.table(df_sorted_r2)
                
                # Migliori modelli
                best_model_mse = df_sorted_mse.iloc[0]["Modello"]
                best_mse = df_sorted_mse.iloc[0]["MSE"]
                
                best_model_r2 = df_sorted_r2.iloc[0]["Modello"]
                best_r2 = df_sorted_r2.iloc[0]["R²"]
                
                st.success(f"Il modello con il miglior MSE è **{best_model_mse}** con un valore di **{best_mse:.4f}**.")
                st.success(f"Il modello con il miglior R² è **{best_model_r2}** con un valore di **{best_r2:.4f}**.")
            else:
                st.info("Non ci sono ancora modelli di regressione addestrati.")

# --- k-Nearest Neighbors (kNN) ---
elif selected_model == "k-Nearest Neighbors (kNN)":
    st.header("k-Nearest Neighbors (kNN)")
    st.markdown("""
    kNN è un algoritmo intuitivo: per classificare un nuovo punto, guarda ai suoi `k` vicini più prossimi nel dataset di addestramento.
    La classe assegnata al nuovo punto è quella più comune tra questi `k` vicini.

    **Configurazione utilizzata:**
    - **k:** √n (radice quadrata del numero di campioni di training) - regola empirica standard
    - **Metrica di distanza:** Euclidea (Minkowski con p=2)
    - **Peso:** Uniforme (tutti i vicini hanno lo stesso peso)
    
    **Caratteristiche:**
    - Algoritmo "lazy": non costruisce un modello esplicito, memorizza tutti i dati di training
    - Sensibile alla scala delle features (i dati sono già standardizzati)
    - Performance dipende fortemente dalla scelta di k
    """)

    if "dataset" not in st.session_state:
        st.warning("Devi prima generare un dataset! Vai alla sezione 'Dataset Generator'.")
    elif st.session_state.dataset["task_type"] != "classification":
        st.error("Questo modello richiede un dataset di classificazione. Torna alla sezione 'Dataset Generator' e crea un nuovo dataset appropriato.")
    else:
        # Estrai il dataset dalla session_state (solo per classificazione)
        X = st.session_state.dataset["X"]
        y = st.session_state.dataset["y"]
        X_train = st.session_state.dataset["X_train"]
        X_test = st.session_state.dataset["X_test"]
        y_train = st.session_state.dataset["y_train"]
        y_test = st.session_state.dataset["y_test"]
        n_features = st.session_state.dataset["n_features"]
        
        st.subheader("k-NN per Classificazione")
        
        # Calcola k ottimale usando la regola empirica
        k_optimal = max(1, int(np.sqrt(len(X_train))))
        
        # Utilizziamo max due features per visualizzazione
        X_knn_train = X_train[:, :min(2, n_features)]
        X_knn_test = X_test[:, :min(2, n_features)]

        # Configurazione standard di kNN
        model_knn = KNeighborsClassifier(n_neighbors=k_optimal)  # Default: weights='uniform', metric='minkowski', p=2
        model_knn.fit(X_knn_train, y_train)
        predictions_knn = model_knn.predict(X_knn_test)
        y_pred_proba_knn = model_knn.predict_proba(X_knn_test)[:, 1]  # Probabilità classe positiva
        
        st.write(f"**Configurazione utilizzata:** k = {k_optimal} (√{len(X_train)} ≈ {np.sqrt(len(X_train)):.1f})")
        st.write(f"**Accuratezza del modello kNN:** {accuracy_score(y_test, predictions_knn):.4f}")

        if n_features >= 2:
            st.markdown("#### Visualizzazione del Modello con Decision Boundary")
            df_test_knn = pd.DataFrame(X_knn_test, columns=['Feature 1', 'Feature 2'])
            df_test_knn['Target'] = y_test

            fig_knn = px.scatter(df_test_knn, x='Feature 1', y='Feature 2', color='Target',
                                 color_discrete_map={0: 'blue', 1: 'red'},
                                 title=f"kNN (k={k_optimal}): Decision Boundary")
            
            x_min_knn, x_max_knn = X[:, 0].min() - .5, X[:, 0].max() + .5
            y_min_knn, y_max_knn = X[:, 1].min() - .5, X[:, 1].max() + .5
            h_knn = .02
            xx_knn, yy_knn = np.meshgrid(np.arange(x_min_knn, x_max_knn, h_knn), np.arange(y_min_knn, y_max_knn, h_knn))
            Z_knn = model_knn.predict(np.c_[xx_knn.ravel(), yy_knn.ravel()])
            Z_knn = Z_knn.reshape(xx_knn.shape)
            contour_colors_knn = [[0, 'lightblue'], [1, 'lightcoral']]
            fig_knn.add_trace(go.Contour(x=xx_knn[0], y=yy_knn[:,0], z=Z_knn, showscale=False, colorscale=contour_colors_knn, opacity=0.4, hoverinfo='skip'))
            st.plotly_chart(fig_knn)

        st.markdown("#### Testa il Modello kNN con Nuovi Dati")
        new_x_values = []
        for i in range(min(2, n_features)):
            new_x_values.append(st.number_input(f"Inserisci valore per Feature {i+1}:", value=float(X[:, i].mean()), format="%.2f", key=f"new_x{i+1}_knn_val"))
        
        if st.button(f"Predici con kNN (k={k_optimal})", key="pred_knn_btn"):
            new_data_knn = np.array([new_x_values])
            pred_class_knn = model_knn.predict(new_data_knn)
            pred_proba_knn = model_knn.predict_proba(new_data_knn)
            st.success(f"Per i valori inseriti:")
            st.write(f"  - Classe Predetta: {pred_class_knn[0]}")
            st.write(f"  - Probabilità (Classe 0): {pred_proba_knn[0][0]:.4f}")
            st.write(f"  - Probabilità (Classe 1): {pred_proba_knn[0][1]:.4f}")
            
        # Salva le metriche di performance
        if "model_performances" not in st.session_state:
            st.session_state.model_performances = {}
            
        st.session_state.model_performances["k-Nearest Neighbors"] = calcola_metriche_classificazione(
            y_test, predictions_knn, y_pred_proba_knn
        )

# --- Decision Trees ---
elif selected_model == "Decision Trees":
    st.header("Alberi Decisionali")
    st.markdown("""
    Gli Alberi Decisionali sono modelli versatili usati sia per **classificazione** che per **regressione**.
    Funzionano creando una struttura ad albero dove:
    - Ogni **nodo interno** rappresenta un "test" su una feature (es. Feature1 < 0.5?)
    - Ogni **ramo** rappresenta l'esito del test
    - Ogni **nodo foglia** rappresenta una classe (per classificazione) o un valore (per regressione)

    **Configurazione utilizzata:**
    - **max_depth = 5:** Profondità massima bilanciata per evitare overfitting mantenendo interpretabilità
    - **Criterio di split:** Gini per classificazione, MSE per regressione (default di scikit-learn)
    - **min_samples_split = 2:** Numero minimo di campioni per dividere un nodo (default)
    
    **Vantaggi:** Altamente interpretabili, gestiscono features numeriche e categoriche, robusti agli outlier
    """)

    if "dataset" not in st.session_state:
        st.warning("Devi prima generare un dataset! Vai alla sezione 'Dataset Generator'.")
    else:
        # Estrai il dataset dalla session_state
        X = st.session_state.dataset["X"]
        y = st.session_state.dataset["y"]
        X_train = st.session_state.dataset["X_train"]
        X_test = st.session_state.dataset["X_test"]
        y_train = st.session_state.dataset["y_train"]
        y_test = st.session_state.dataset["y_test"]
        task_type = st.session_state.dataset["task_type"]
        n_features = st.session_state.dataset["n_features"]

        # Configurazione standard bilanciata
        max_depth_standard = 5

        if task_type == "classification":
            st.subheader("Albero Decisionale per Classificazione")
            
            # Utilizziamo max due features per visualizzazione
            X_tree_train = X_train[:, :min(2, n_features)]
            X_tree_test = X_test[:, :min(2, n_features)]

            model_tree_clf = DecisionTreeClassifier(max_depth=max_depth_standard, random_state=42)
            model_tree_clf.fit(X_tree_train, y_train)
            predictions_tc = model_tree_clf.predict(X_tree_test)
            y_pred_proba_tc = model_tree_clf.predict_proba(X_tree_test)[:, 1]

            st.write(f"**Accuratezza (max_depth={max_depth_standard}):** {accuracy_score(y_test, predictions_tc):.4f}")
            st.write(f"**Profondità effettiva dell'albero:** {model_tree_clf.get_depth()}")
            st.write(f"**Numero di nodi foglia:** {model_tree_clf.get_n_leaves()}")

            st.markdown("#### Visualizzazione dell'Albero Decisionale")
            fig_tree_plot_clf, ax_clf = plt.subplots(figsize=(12, 8))
            feature_names = [f'Feature {i+1}' for i in range(min(2, n_features))]
            plot_tree(model_tree_clf, filled=True, feature_names=feature_names, class_names=['Classe 0', 'Classe 1'], rounded=True, ax=ax_clf)
            st.pyplot(fig_tree_plot_clf)

            if n_features >= 2:
                st.markdown("#### Visualizzazione della Decision Boundary")
                df_test_tc = pd.DataFrame(X_tree_test, columns=['Feature 1', 'Feature 2'])
                df_test_tc['Target'] = y_test
                fig_tc_boundary = px.scatter(df_test_tc, x='Feature 1', y='Feature 2', color='Target',
                                           color_discrete_map={0: 'blue', 1: 'red'},
                                           title=f"Decision Tree (max_depth={max_depth_standard}): Decision Boundary")
                
                x_min_tc, x_max_tc = X[:, 0].min() - .5, X[:, 0].max() + .5
                y_min_tc, y_max_tc = X[:, 1].min() - .5, X[:, 1].max() + .5
                h_tc = .02
                xx_tc, yy_tc = np.meshgrid(np.arange(x_min_tc, x_max_tc, h_tc), np.arange(y_min_tc, y_max_tc, h_tc))
                Z_tc = model_tree_clf.predict(np.c_[xx_tc.ravel(), yy_tc.ravel()])
                Z_tc = Z_tc.reshape(xx_tc.shape)
                contour_colors_tc = [[0, 'lightblue'], [1, 'lightcoral']]
                fig_tc_boundary.add_trace(go.Contour(x=xx_tc[0], y=yy_tc[:,0], z=Z_tc, showscale=False, colorscale=contour_colors_tc, opacity=0.4, hoverinfo='skip'))
                st.plotly_chart(fig_tc_boundary)

            st.markdown("#### Testa l'Albero di Classificazione con Nuovi Dati")
            new_x_values = []
            for i in range(min(2, n_features)):
                new_x_values.append(st.number_input(f"Inserisci valore per Feature {i+1}:", value=float(X[:, i].mean()), format="%.2f", key=f"new_x{i+1}_tc_val"))
            
            if st.button("Predici con Decision Tree", key="pred_tc_btn"):
                new_data_tc = np.array([new_x_values])
                pred_class_tc = model_tree_clf.predict(new_data_tc)
                pred_proba_tc = model_tree_clf.predict_proba(new_data_tc)
                st.success(f"Per i valori inseriti:")
                st.write(f"  - Classe Predetta: {pred_class_tc[0]}")
                st.write(f"  - Probabilità (Classe 0): {pred_proba_tc[0][0]:.4f}")
                st.write(f"  - Probabilità (Classe 1): {pred_proba_tc[0][1]:.4f}")
                
            # Salva le metriche di performance
            if "model_performances" not in st.session_state:
                st.session_state.model_performances = {}
                
            st.session_state.model_performances["Decision Tree"] = calcola_metriche_classificazione(
                y_test, predictions_tc, y_pred_proba_tc
            )

        elif task_type == "regression":
            st.subheader("Albero Decisionale per Regressione")
            
            # Utilizziamo solo la prima feature per visualizzazione semplice
            X_tree_train = X_train[:, 0].reshape(-1, 1)
            X_tree_test = X_test[:, 0].reshape(-1, 1)

            model_tree_reg = DecisionTreeRegressor(max_depth=max_depth_standard, random_state=42)
            model_tree_reg.fit(X_tree_train, y_train)
            predictions_tr = model_tree_reg.predict(X_tree_test)
            mse_tr = mean_squared_error(y_test, predictions_tr)
            r2_tr = r2_score(y_test, predictions_tr)

            st.write(f"**Mean Squared Error (MSE):** {mse_tr:.4f}")
            st.write(f"**R² Score:** {r2_tr:.4f}")
            st.write(f"**Profondità effettiva dell'albero:** {model_tree_reg.get_depth()}")
            st.write(f"**Numero di nodi foglia:** {model_tree_reg.get_n_leaves()}")

            st.markdown("#### Visualizzazione dell'Albero Decisionale")
            fig_tree_plot_reg, ax_reg = plt.subplots(figsize=(12, 8))
            plot_tree(model_tree_reg, filled=True, feature_names=['Feature 1'], rounded=True, ax=ax_reg, precision=2)
            st.pyplot(fig_tree_plot_reg)

            st.markdown("#### Visualizzazione della Funzione Appresa")
            fig_tr_func = go.Figure()
            fig_tr_func.add_trace(go.Scatter(x=X_tree_test[:,0], y=y_test, mode='markers', name='Dati di Test'))
            x_range_tr = np.sort(X_tree_test[:, 0])
            y_range_tr = model_tree_reg.predict(x_range_tr.reshape(-1, 1))
            fig_tr_func.add_trace(go.Scatter(x=x_range_tr, y=y_range_tr, mode='lines', name=f'Decision Tree (max_depth={max_depth_standard})'))
            fig_tr_func.update_layout(title=f"Decision Tree Regressione: Funzione Appresa", xaxis_title="Feature", yaxis_title="Target")
            st.plotly_chart(fig_tr_func)

            st.markdown("#### Testa l'Albero di Regressione con un Nuovo Dato")
            new_x_tr = st.number_input("Valore Feature:", value=float(X[:, 0].mean()), format="%.2f", key="new_x_tr")
            if st.button("Predici con Decision Tree", key="pred_tr_btn"):
                predicted_y_tr = model_tree_reg.predict(np.array([[new_x_tr]]))
                st.success(f"Per Feature = {new_x_tr:.2f}, il valore predetto è: {predicted_y_tr[0]:.4f}")
                
            # Salva le metriche di performance
            if "model_performances" not in st.session_state:
                st.session_state.model_performances = {}
                
            st.session_state.model_performances["Decision Tree"] = {
                "mse": mse_tr,
                "r2": r2_tr
            }
        else:
            st.error("Tipo di task non riconosciuto. Genera un nuovo dataset.")

# --- Neural Networks (MLP) ---
elif selected_model == "Neural Networks":
    st.header("Reti Neurali (Multi-layer Perceptron - MLP)")
    st.markdown("""
    Le Reti Neurali Artificiali sono modelli ispirati al cervello umano, composti da "neuroni" organizzati in "layer" (strati).
    Un Multi-layer Perceptron (MLP) è un tipo comune di rete neurale feedforward.
    - **Input Layer:** Riceve le features.
    - **Hidden Layers:** Strati intermedi dove avvengono le computazioni. Ogni neurone applica una trasformazione lineare seguita da una **funzione di attivazione** (es. ReLU, sigmoide) che introduce non-linearità, permettendo di apprendere relazioni complesse.
    - **Output Layer:** Produce la predizione (es. probabilità di classe per classificazione, valore numerico per regressione).
    
    L'addestramento avviene tipicamente tramite **backpropagation** e algoritmi di ottimizzazione come Adam o SGD per aggiustare i "pesi" della rete.

    **Interpretabilità:** Le reti neurali profonde sono spesso considerate "scatole nere" (black box) perché può essere difficile interpretare esattamente cosa ogni neurone o peso rappresenti. Tuttavia, la loro architettura (numero di layer, neuroni per layer, funzioni di attivazione) è cruciale per le loro performance.
    Useremo una struttura MLP semplice per questo esempio.
    """)

    if "dataset" not in st.session_state:
        st.warning("Devi prima generare un dataset! Vai alla sezione 'Dataset Generator'.")
    else:
        # Estrai il dataset dalla session_state
        X = st.session_state.dataset["X"]
        y = st.session_state.dataset["y"]
        X_train = st.session_state.dataset["X_train"]
        X_test = st.session_state.dataset["X_test"]
        y_train = st.session_state.dataset["y_train"]
        y_test = st.session_state.dataset["y_test"]
        task_type = st.session_state.dataset["task_type"]
        n_features = st.session_state.dataset["n_features"]

        nn_task_type = st.selectbox("Scegli il tipo di task per la Rete Neurale:", ["Classificazione", "Regressione"], key="nn_task_simplified")
        
        # Parametri MLP fissi per semplicità
        # Struttura Hidden Layer: un layer con 50 neuroni. Puoi cambiarlo per sperimentare.
        # Es: (50,) -> un layer da 50 neuroni
        # Es: (20,10) -> due layer, uno da 20 e uno da 10 neuroni
        hidden_layer_config = (50,)
        activation_func = 'relu'
        solver_type = 'adam'
        max_iterations = 500 

        st.info(f"Configurazione Rete Neurale usata: hidden_layer_sizes={hidden_layer_config}, activation='{activation_func}', solver='{solver_type}', max_iter={max_iterations}. La convergenza potrebbe richiedere qualche secondo.")

        if nn_task_type == "Classificazione" and task_type == "classification":
            st.subheader("MLP per Classificazione")
            
            # Utilizziamo max due features per visualizzazione
            X_nn_train = X_train[:, :min(2, n_features)]
            X_nn_test = X_test[:, :min(2, n_features)]

            model_nn_clf = MLPClassifier(hidden_layer_sizes=hidden_layer_config, activation=activation_func, solver=solver_type, max_iter=max_iterations, random_state=42)
            model_nn_clf.fit(X_nn_train, y_train)
            predictions_nnc = model_nn_clf.predict(X_nn_test)
            y_pred_proba_nnc = model_nn_clf.predict_proba(X_nn_test)[:, 1]

            st.write(f"**Accuratezza (MLP Classifier):** {accuracy_score(y_test, predictions_nnc):.4f}")

            if n_features >= 2:
                st.markdown("#### Visualizzazione della Decision Boundary (MLP Classificazione)")
                df_test_nnc = pd.DataFrame(X_nn_test, columns=['Feature 1', 'Feature 2'])
                df_test_nnc['Target'] = y_test
                fig_nnc_boundary = px.scatter(df_test_nnc, x='Feature 1', y='Feature 2', color='Target',
                                           color_discrete_map={0: 'blue', 1: 'red'},
                                           title=f"MLP Classificazione: Decision Boundary")
                
                x_min_nnc, x_max_nnc = X[:, 0].min() - .5, X[:, 0].max() + .5
                y_min_nnc, y_max_nnc = X[:, 1].min() - .5, X[:, 1].max() + .5
                h_nnc = .02 # Aumentare per performance, diminuire per risoluzione
                xx_nnc, yy_nnc = np.meshgrid(np.arange(x_min_nnc, x_max_nnc, h_nnc), np.arange(y_min_nnc, y_max_nnc, h_nnc))
                Z_nnc = model_nn_clf.predict(np.c_[xx_nnc.ravel(), yy_nnc.ravel()])
                Z_nnc = Z_nnc.reshape(xx_nnc.shape)
                contour_colors_nnc = [[0, 'lightblue'], [1, 'lightcoral']]
                fig_nnc_boundary.add_trace(go.Contour(x=xx_nnc[0], y=yy_nnc[:,0], z=Z_nnc, showscale=False, colorscale=contour_colors_nnc, opacity=0.4, hoverinfo='skip'))
                st.plotly_chart(fig_nnc_boundary)

            st.markdown("#### Testa l'MLP di Classificazione con Nuovi Dati")
            new_x_values = []
            for i in range(min(2, n_features)):
                new_x_values.append(st.number_input(f"Inserisci valore per Feature {i+1}:", value=float(X[:, i].mean()), format="%.2f", key=f"new_x{i+1}_nnc_val"))
            
            if st.button("Predici con MLP Classificazione", key="pred_nnc_btn"):
                new_data_nnc = np.array([new_x_values])
                pred_class_nnc = model_nn_clf.predict(new_data_nnc)
                pred_proba_nnc = model_nn_clf.predict_proba(new_data_nnc)
                st.success(f"Per i valori inseriti:")
                st.write(f"  - Classe Predetta: {pred_class_nnc[0]}")
                st.write(f"  - Probabilità (Classe 0): {pred_proba_nnc[0][0]:.4f}")
                st.write(f"  - Probabilità (Classe 1): {pred_proba_nnc[0][1]:.4f}")
                
            # Salva le metriche di performance
            if "model_performances" not in st.session_state:
                st.session_state.model_performances = {}
                
            st.session_state.model_performances["Neural Network (Classificazione)"] = calcola_metriche_classificazione(
                y_test, predictions_nnc, y_pred_proba_nnc
            )

        elif nn_task_type == "Regressione" and task_type == "regression":
            st.subheader("MLP per Regressione")
            
            # Utilizziamo solo la prima feature per visualizzazione semplice
            X_nn_train = X_train[:, 0].reshape(-1, 1)
            X_nn_test = X_test[:, 0].reshape(-1, 1)

            model_nn_reg = MLPRegressor(hidden_layer_sizes=hidden_layer_config, activation=activation_func, solver=solver_type, max_iter=max_iterations, random_state=42)
            model_nn_reg.fit(X_nn_train, y_train)
            predictions_nnr = model_nn_reg.predict(X_nn_test)
            mse_nnr = mean_squared_error(y_test, predictions_nnr)
            r2_nnr = r2_score(y_test, predictions_nnr)

            st.write(f"**Mean Squared Error (MSE) (MLP Regressor):** {mse_nnr:.4f}")
            st.write(f"**R² Score (MLP Regressor):** {r2_nnr:.4f}")

            st.markdown("#### Visualizzazione della Funzione Appresa (MLP Regressione)")
            fig_nnr_func = go.Figure()
            fig_nnr_func.add_trace(go.Scatter(x=X_nn_test[:,0], y=y_test, mode='markers', name='Dati di Test'))
            x_range_nnr = np.sort(X_nn_test[:, 0]) 
            y_range_nnr = model_nn_reg.predict(x_range_nnr.reshape(-1, 1))
            fig_nnr_func.add_trace(go.Scatter(x=x_range_nnr, y=y_range_nnr, mode='lines', name='Predizione MLP'))
            fig_nnr_func.update_layout(title="MLP Regressione: Funzione Appresa", xaxis_title="Feature", yaxis_title="Target")
            st.plotly_chart(fig_nnr_func)

            st.markdown("#### Testa l'MLP di Regressione con un Nuovo Dato")
            new_x_nnr = st.number_input("Valore Feature:", value=float(X[:, 0].mean()), format="%.2f", key="new_x_nnr")
            if st.button("Predici con MLP Regressione", key="pred_nnr_btn"):
                predicted_y_nnr = model_nn_reg.predict(np.array([[new_x_nnr]]))
                st.success(f"Per Feature = {new_x_nnr:.2f}, il valore predetto è: {predicted_y_nnr[0]:.4f}")
                
            # Salva le metriche di performance
            if "model_performances" not in st.session_state:
                st.session_state.model_performances = {}
                
            st.session_state.model_performances["Neural Network (Regressione)"] = {
                "mse": mse_nnr,
                "r2": r2_nnr
            }
        else:
            st.error(f"Incompatibilità tra il tipo di task del dataset ({task_type}) e il tipo di rete neurale selezionato ({nn_task_type}). Crea un nuovo dataset appropriato o seleziona un tipo di task compatibile.")

# --- Random Forest ---
elif selected_model == "Random Forest":
    st.header("Random Forest")
    st.markdown("""
    Il **Random Forest** è un potente algoritmo di ensemble learning che costruisce una "foresta" di molteplici alberi decisionali.
    
    **Principio di funzionamento:**
    1. **Bootstrap Sampling:** Crea sotto-dataset campionando casualmente con rimpiazzo
    2. **Selezione casuale delle features:** Ad ogni split, considera solo un sottoinsieme casuale delle features
    3. **Aggregazione:** Combina le predizioni di tutti gli alberi (voto maggioranza per classificazione, media per regressione)
    
    **Configurazione utilizzata:**
    - **n_estimators = 100:** Numero di alberi nella foresta (default ottimale)
    - **max_features = 'sqrt':** √(n_features) features considerate ad ogni split (default per classificazione)
    - **max_depth = None:** Gli alberi crescono fino a foglie pure (default)
    - **bootstrap = True:** Utilizza bootstrap sampling (default)
    
    **Vantaggi:** Riduce overfitting, fornisce importanza delle features, robusto agli outlier, parallelizzabile
    """)

    if "dataset" not in st.session_state:
        st.warning("Devi prima generare un dataset! Vai alla sezione 'Dataset Generator'.")
    else:
        # Estrai il dataset dalla session_state
        X = st.session_state.dataset["X"]
        y = st.session_state.dataset["y"]
        X_train = st.session_state.dataset["X_train"]
        X_test = st.session_state.dataset["X_test"]
        y_train = st.session_state.dataset["y_train"]
        y_test = st.session_state.dataset["y_test"]
        task_type = st.session_state.dataset["task_type"]
        n_features = st.session_state.dataset["n_features"]

        # Configurazione standard Random Forest
        n_estimators_rf = 100

        if task_type == "classification":
            st.subheader("Random Forest per Classificazione")
            
            # Utilizziamo max due features per visualizzazione
            X_rf_train = X_train[:, :min(2, n_features)]
            X_rf_test = X_test[:, :min(2, n_features)]

            model_rf_clf = RandomForestClassifier(n_estimators=n_estimators_rf, random_state=42)
            model_rf_clf.fit(X_rf_train, y_train)
            predictions_rfc = model_rf_clf.predict(X_rf_test)
            y_pred_proba_rfc = model_rf_clf.predict_proba(X_rf_test)[:, 1]

            st.write(f"**Accuratezza:** {accuracy_score(y_test, predictions_rfc):.4f}")
            st.write(f"**Numero di alberi:** {n_estimators_rf}")
            
            if hasattr(model_rf_clf, 'feature_importances_'):
                importances = model_rf_clf.feature_importances_
                feature_names = [f'Feature {i+1}' for i in range(min(2, n_features))]
                st.write("**Importanza delle Features:**")
                df_importances = pd.DataFrame({'Feature': feature_names, 'Importanza': importances[:len(feature_names)]}).sort_values(by='Importanza', ascending=False)
                st.dataframe(df_importances.style.format({"Importanza": "{:.4f}"}))

            if n_features >= 2:
                st.markdown("#### Visualizzazione della Decision Boundary")
                df_test_rfc = pd.DataFrame(X_rf_test, columns=['Feature 1', 'Feature 2'])
                df_test_rfc['Target'] = y_test
                fig_rfc_boundary = px.scatter(df_test_rfc, x='Feature 1', y='Feature 2', color='Target',
                                           color_discrete_map={0: 'blue', 1: 'red'},
                                           title="Random Forest: Decision Boundary")
                
                x_min_rfc, x_max_rfc = X[:, 0].min() - .5, X[:, 0].max() + .5
                y_min_rfc, y_max_rfc = X[:, 1].min() - .5, X[:, 1].max() + .5
                h_rfc = .02
                xx_rfc, yy_rfc = np.meshgrid(np.arange(x_min_rfc, x_max_rfc, h_rfc), np.arange(y_min_rfc, y_max_rfc, h_rfc))
                Z_rfc = model_rf_clf.predict(np.c_[xx_rfc.ravel(), yy_rfc.ravel()])
                Z_rfc = Z_rfc.reshape(xx_rfc.shape)
                contour_colors_rfc = [[0, 'lightblue'], [1, 'lightcoral']]
                fig_rfc_boundary.add_trace(go.Contour(x=xx_rfc[0], y=yy_rfc[:,0], z=Z_rfc, showscale=False, colorscale=contour_colors_rfc, opacity=0.4, hoverinfo='skip'))
                st.plotly_chart(fig_rfc_boundary)

            st.markdown("#### Testa Random Forest con Nuovi Dati")
            new_x_values = []
            for i in range(min(2, n_features)):
                new_x_values.append(st.number_input(f"Inserisci valore per Feature {i+1}:", value=float(X[:, i].mean()), format="%.2f", key=f"new_x{i+1}_rfc_val"))
            
            if st.button("Predici con Random Forest", key="pred_rfc_btn"):
                new_data_rfc = np.array([new_x_values])
                pred_class_rfc = model_rf_clf.predict(new_data_rfc)
                pred_proba_rfc = model_rf_clf.predict_proba(new_data_rfc)
                st.success(f"Per i valori inseriti:")
                st.write(f"  - Classe Predetta: {pred_class_rfc[0]}")
                st.write(f"  - Probabilità (Classe 0): {pred_proba_rfc[0][0]:.4f}")
                st.write(f"  - Probabilità (Classe 1): {pred_proba_rfc[0][1]:.4f}")
                
            # Salva le metriche di performance
            if "model_performances" not in st.session_state:
                st.session_state.model_performances = {}
                
            st.session_state.model_performances["Random Forest"] = calcola_metriche_classificazione(
                y_test, predictions_rfc, y_pred_proba_rfc
            )

        elif task_type == "regression":
            st.subheader("Random Forest per Regressione")
            
            # Importazione specifica per Random Forest Regressor
            from sklearn.ensemble import RandomForestRegressor
            
            # Utilizziamo solo la prima feature per visualizzazione semplice
            X_rf_train = X_train[:, 0].reshape(-1, 1)
            X_rf_test = X_test[:, 0].reshape(-1, 1)

            model_rf_reg = RandomForestRegressor(n_estimators=n_estimators_rf, random_state=42)
            model_rf_reg.fit(X_rf_train, y_train)
            predictions_rfr = model_rf_reg.predict(X_rf_test)
            mse_rfr = mean_squared_error(y_test, predictions_rfr)
            r2_rfr = r2_score(y_test, predictions_rfr)

            st.write(f"**Mean Squared Error (MSE):** {mse_rfr:.4f}")
            st.write(f"**R² Score:** {r2_rfr:.4f}")
            st.write(f"**Numero di alberi:** {n_estimators_rf}")

            st.markdown("#### Visualizzazione della Funzione Appresa")
            fig_rfr_func = go.Figure()
            fig_rfr_func.add_trace(go.Scatter(x=X_rf_test[:,0], y=y_test, mode='markers', name='Dati di Test'))
            x_range_rfr = np.sort(X_rf_test[:, 0]) 
            y_range_rfr = model_rf_reg.predict(x_range_rfr.reshape(-1, 1))
            fig_rfr_func.add_trace(go.Scatter(x=x_range_rfr, y=y_range_rfr, mode='lines', name='Random Forest'))
            fig_rfr_func.update_layout(title="Random Forest Regressione: Funzione Appresa", xaxis_title="Feature", yaxis_title="Target")
            st.plotly_chart(fig_rfr_func)

            st.markdown("#### Testa Random Forest con un Nuovo Dato")
            new_x_rfr = st.number_input("Valore Feature:", value=float(X[:, 0].mean()), format="%.2f", key="new_x_rfr")
            if st.button("Predici con Random Forest", key="pred_rfr_btn"):
                predicted_y_rfr = model_rf_reg.predict(np.array([[new_x_rfr]]))
                st.success(f"Per Feature = {new_x_rfr:.2f}, il valore predetto è: {predicted_y_rfr[0]:.4f}")
                
            # Salva le metriche di performance
            if "model_performances" not in st.session_state:
                st.session_state.model_performances = {}
                
            st.session_state.model_performances["Random Forest"] = {
                "mse": mse_rfr,
                "r2": r2_rfr
            }
        else:
            st.error("Tipo di task non riconosciuto. Genera un nuovo dataset.")

# --- Support Vector Machines (SVM) ---
elif selected_model == "Support Vector Machines (SVM)":
    st.header("Support Vector Machines (SVM)")
    st.markdown("""
    Le Support Vector Machines (SVM) sono algoritmi potenti sia per **classificazione** che per **regressione**.
    Principi fondamentali:
    - **Classificazione:** Cerca l'iperpiano ottimale che separa le classi con il massimo margine.
    - **Regressione (SVR):** Cerca una funzione con al massimo ε di deviazione dai valori target, minimizzando la complessità.
    - **Kernel Trick:** Permette di gestire dati non linearmente separabili mappandoli in uno spazio di dimensione superiore.
    
    **Kernel comuni:**
    - **Lineare:** K(x, y) = x · y (prodotto scalare). Semplice ed efficace per dati linearmente separabili.
    - **RBF (Radial Basis Function):** K(x, y) = exp(-γ||x-y||²). Gestisce relazioni non lineari, adatto a molte situazioni.
    - **Polinomiale:** K(x, y) = (γx · y + r)^d. Utile per dati con relazioni di ordine superiore.
    
    **Parametri importanti:**
    - **C:** Controlla il trade-off tra margine e errori di classificazione. Un valore basso di C crea un margine più ampio ma permette più errori.
    - **gamma (γ):** (Per kernel non lineari) Definisce quanto lontano arriva l'influenza di un singolo esempio. Valori alti = influenza più locale.
    """)

    if "dataset" not in st.session_state:
        st.warning("Devi prima generare un dataset! Vai alla sezione 'Dataset Generator'.")
    else:
        # Estrai il dataset dalla session_state
        X = st.session_state.dataset["X"]
        y = st.session_state.dataset["y"]
        X_train = st.session_state.dataset["X_train"]
        X_test = st.session_state.dataset["X_test"]
        y_train = st.session_state.dataset["y_train"]
        y_test = st.session_state.dataset["y_test"]
        task_type = st.session_state.dataset["task_type"]
        n_features = st.session_state.dataset["n_features"]
        
        # Importazioni per SVM
        from sklearn.svm import SVC, SVR

        svm_task_type = st.selectbox("Scegli il tipo di task per SVM:", ["Classificazione", "Regressione"], key="svm_task_simplified")
        kernel_type = st.selectbox("Scegli il tipo di kernel:", ["linear", "rbf", "poly"], key="svm_kernel")
        C_param = st.slider("Parametro di regolarizzazione C:", 0.1, 10.0, 1.0, 0.1, key="svm_c_param")
        
        if kernel_type != "linear":
            gamma_param = st.select_slider("Parametro gamma:", options=["scale", "auto", 0.01, 0.1, 1.0, 10.0], value="scale", key="svm_gamma")
            if gamma_param in ["0.01", "0.1", "1.0", "10.0"]:
                gamma_param = float(gamma_param)
        else:
            gamma_param = "scale"  # Default per kernel lineare

        if svm_task_type == "Classificazione" and task_type == "classification":
            st.subheader("SVM per Classificazione")
            
            # Utilizziamo max due features per visualizzazione
            X_svm_train = X_train[:, :min(2, n_features)]
            X_svm_test = X_test[:, :min(2, n_features)]

            model_svm_clf = SVC(kernel=kernel_type, C=C_param, gamma=gamma_param, probability=True, random_state=42)
            model_svm_clf.fit(X_svm_train, y_train)
            predictions_svm = model_svm_clf.predict(X_svm_test)
            y_pred_proba_svm = model_svm_clf.predict_proba(X_svm_test)[:, 1]

            st.write(f"**Accuratezza (SVM Classifier, kernel={kernel_type}, C={C_param}):** {accuracy_score(y_test, predictions_svm):.4f}")
            
            # Informazioni aggiuntive
            n_support_vectors = model_svm_clf.n_support_.sum()
            st.write(f"**Numero totale di vettori di supporto:** {n_support_vectors}")
            st.write(f"*I vettori di supporto sono i punti più vicini all'iperpiano decisionale e determinano il margine.*")

            if n_features >= 2:
                st.markdown("#### Visualizzazione della Decision Boundary (SVM Classificazione)")
                df_test_svm = pd.DataFrame(X_svm_test, columns=['Feature 1', 'Feature 2'])
                df_test_svm['Target'] = y_test
                fig_svm_boundary = px.scatter(df_test_svm, x='Feature 1', y='Feature 2', color='Target',
                                           color_discrete_map={0: 'blue', 1: 'red'},
                                           title=f"SVM Classificazione (kernel={kernel_type}, C={C_param}): Decision Boundary")
                
                x_min_svm, x_max_svm = X[:, 0].min() - .5, X[:, 0].max() + .5
                y_min_svm, y_max_svm = X[:, 1].min() - .5, X[:, 1].max() + .5
                h_svm = .02
                xx_svm, yy_svm = np.meshgrid(np.arange(x_min_svm, x_max_svm, h_svm), np.arange(y_min_svm, y_max_svm, h_svm))
                Z_svm = model_svm_clf.predict(np.c_[xx_svm.ravel(), yy_svm.ravel()])
                Z_svm = Z_svm.reshape(xx_svm.shape)
                contour_colors_svm = [[0, 'lightblue'], [1, 'lightcoral']]
                fig_svm_boundary.add_trace(go.Contour(x=xx_svm[0], y=yy_svm[:,0], z=Z_svm, showscale=False, colorscale=contour_colors_svm, opacity=0.4, hoverinfo='skip'))
                st.plotly_chart(fig_svm_boundary)

            st.markdown("#### Testa SVM Classifier con Nuovi Dati")
            new_x_values = []
            for i in range(min(2, n_features)):
                new_x_values.append(st.number_input(f"Inserisci valore per Feature {i+1}:", value=float(X[:, i].mean()), format="%.2f", key=f"new_x{i+1}_svm_val"))
            
            if st.button("Predici con SVM Classifier", key="pred_svm_btn"):
                new_data_svm = np.array([new_x_values])
                pred_class_svm = model_svm_clf.predict(new_data_svm)
                pred_proba_svm = model_svm_clf.predict_proba(new_data_svm)
                st.success(f"Per i valori inseriti:")
                st.write(f"  - Classe Predetta: {pred_class_svm[0]}")
                st.write(f"  - Probabilità (Classe 0): {pred_proba_svm[0][0]:.4f}")
                st.write(f"  - Probabilità (Classe 1): {pred_proba_svm[0][1]:.4f}")
                
            # Salva le metriche di performance
            if "model_performances" not in st.session_state:
                st.session_state.model_performances = {}
                
            st.session_state.model_performances[f"SVM (Classificazione, {kernel_type})"] = calcola_metriche_classificazione(
                y_test, predictions_svm, y_pred_proba_svm
            )

        elif svm_task_type == "Regressione" and task_type == "regression":
            st.subheader("SVM per Regressione (SVR)")
            
            # Utilizziamo solo la prima feature per visualizzazione semplice
            X_svm_train = X_train[:, 0].reshape(-1, 1)
            X_svm_test = X_test[:, 0].reshape(-1, 1)

            model_svm_reg = SVR(kernel=kernel_type, C=C_param, gamma=gamma_param)
            model_svm_reg.fit(X_svm_train, y_train)
            predictions_svr = model_svm_reg.predict(X_svm_test)
            mse_svr = mean_squared_error(y_test, predictions_svr)
            r2_svr = r2_score(y_test, predictions_svr)

            st.write(f"**Mean Squared Error (MSE) (SVR, kernel={kernel_type}, C={C_param}):** {mse_svr:.4f}")
            st.write(f"**R² Score (SVR, kernel={kernel_type}):** {r2_svr:.4f}")
            
            # Informazioni aggiuntive
            n_support_vectors_reg = model_svm_reg.support_.shape[0]
            st.write(f"**Numero di vettori di supporto:** {n_support_vectors_reg}")
            st.write(f"*I vettori di supporto sono i punti che definiscono la funzione di regressione.*")

            st.markdown("#### Visualizzazione della Funzione Appresa (SVR)")
            fig_svr_func = go.Figure()
            fig_svr_func.add_trace(go.Scatter(x=X_svm_test[:,0], y=y_test, mode='markers', name='Dati di Test'))
            x_range_svr = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
            y_range_svr = model_svm_reg.predict(x_range_svr.reshape(-1, 1))
            fig_svr_func.add_trace(go.Scatter(x=x_range_svr, y=y_range_svr, mode='lines', name=f'Predizione SVR ({kernel_type})'))
            fig_svr_func.update_layout(title=f"SVR (kernel={kernel_type}, C={C_param}): Funzione Appresa", xaxis_title="Feature", yaxis_title="Target")
            st.plotly_chart(fig_svr_func)

            st.markdown("#### Testa SVR con un Nuovo Dato")
            new_x_svr = st.number_input("Valore Feature:", value=float(X[:, 0].mean()), format="%.2f", key="new_x_svr")
            if st.button("Predici con SVR", key="pred_svr_btn"):
                predicted_y_svr = model_svm_reg.predict(np.array([[new_x_svr]]))
                st.success(f"Per Feature = {new_x_svr:.2f}, il valore predetto è: {predicted_y_svr[0]:.4f}")
                
            # Salva le metriche di performance
            if "model_performances" not in st.session_state:
                st.session_state.model_performances = {}
                
            st.session_state.model_performances[f"SVM (Regressione, {kernel_type})"] = {
                "mse": mse_svr,
                "r2": r2_svr
            }
        else:
            st.error(f"Incompatibilità tra il tipo di task del dataset ({task_type}) e il tipo di SVM selezionato ({svm_task_type}). Crea un nuovo dataset appropriato o seleziona un tipo di task compatibile.")

# --- Ensemble Methods: Bagging ---
elif selected_model == "Ensemble Methods: Bagging":
    st.header("Ensemble Methods: Bagging")
    st.markdown("""
    Il **Bagging** (Bootstrap Aggregating) è una tecnica di ensemble learning che mira a migliorare la stabilità e l'accuratezza dei modelli di machine learning, riducendone la varianza e aiutando a evitare l'overfitting.

    Come funziona:
    1.  **Bootstrap Sampling:** Vengono creati multipli sotto-dataset campionando casualmente dal dataset di training originale *con rimpiazzo*. Ogni sotto-dataset ha la stessa dimensione dell'originale.
    2.  **Addestramento Modelli Base:** Un modello base (spesso un Albero Decisionale) viene addestrato indipendentemente su ciascuno di questi sotto-dataset.
    3.  **Aggregazione:** Le predizioni di tutti i modelli base vengono aggregate:
        -   Per la **classificazione**, si usa tipicamente il voto di maggioranza.
        -   Per la **regressione**, si usa tipicamente la media delle predizioni.

    L'idea è che, combinando le "opinioni" di diversi modelli addestrati su versioni leggermente diverse dei dati, si ottiene una predizione complessiva più robusta.
    Useremo Alberi Decisionali (con `max_depth=3`) come modelli base e `n_estimators=10` (numero di alberi).
    """)

    if "dataset" not in st.session_state:
        st.warning("Devi prima generare un dataset! Vai alla sezione 'Dataset Generator'.")
    else:
        # Estrai il dataset dalla session_state
        X = st.session_state.dataset["X"]
        y = st.session_state.dataset["y"]
        X_train = st.session_state.dataset["X_train"]
        X_test = st.session_state.dataset["X_test"]
        y_train = st.session_state.dataset["y_train"]
        y_test = st.session_state.dataset["y_test"]
        task_type = st.session_state.dataset["task_type"]
        n_features = st.session_state.dataset["n_features"]

        bagging_task_type = st.selectbox("Scegli il tipo di task per il Bagging:", ["Classificazione", "Regressione"], key="bagging_task_simplified")
        
        # Parametri Bagging fissi per semplicità
        n_estimators_bagging = 10
        base_estimator_max_depth = 3 # Per rendere gli alberi base non troppo complessi

        st.info(f"Configurazione Bagging: n_estimators={n_estimators_bagging}, base_estimator=DecisionTree(max_depth={base_estimator_max_depth}).")

        if bagging_task_type == "Classificazione" and task_type == "classification":
            st.subheader("Bagging per Classificazione")
            
            # Utilizziamo max due features per visualizzazione
            X_bag_train = X_train[:, :min(2, n_features)]
            X_bag_test = X_test[:, :min(2, n_features)]

            base_clf = DecisionTreeClassifier(max_depth=base_estimator_max_depth, random_state=42)
            model_bag_clf = BaggingClassifier(estimator=base_clf, n_estimators=n_estimators_bagging, random_state=42)
            model_bag_clf.fit(X_bag_train, y_train)
            predictions_bc = model_bag_clf.predict(X_bag_test)
            y_pred_proba_bc = model_bag_clf.predict_proba(X_bag_test)[:, 1]

            st.write(f"**Accuratezza (Bagging Classifier):** {accuracy_score(y_test, predictions_bc):.4f}")

            if n_features >= 2:
                st.markdown("#### Visualizzazione della Decision Boundary (Bagging Classificazione)")
                df_test_bc = pd.DataFrame(X_bag_test, columns=['Feature 1', 'Feature 2'])
                df_test_bc['Target'] = y_test
                fig_bc_boundary = px.scatter(df_test_bc, x='Feature 1', y='Feature 2', color='Target',
                                          color_discrete_map={0: 'blue', 1: 'red'},
                                          title=f"Bagging Classificazione: Decision Boundary")
                
                x_min_bc, x_max_bc = X[:, 0].min() - .5, X[:, 0].max() + .5
                y_min_bc, y_max_bc = X[:, 1].min() - .5, X[:, 1].max() + .5
                h_bc = .02 
                xx_bc, yy_bc = np.meshgrid(np.arange(x_min_bc, x_max_bc, h_bc), np.arange(y_min_bc, y_max_bc, h_bc))
                Z_bc = model_bag_clf.predict(np.c_[xx_bc.ravel(), yy_bc.ravel()])
                Z_bc = Z_bc.reshape(xx_bc.shape)
                contour_colors_bc = [[0, 'lightblue'], [1, 'lightcoral']]
                fig_bc_boundary.add_trace(go.Contour(x=xx_bc[0], y=yy_bc[:,0], z=Z_bc, showscale=False, colorscale=contour_colors_bc, opacity=0.4, hoverinfo='skip'))
                st.plotly_chart(fig_bc_boundary)

            st.markdown("#### Testa il Bagging Classifier con Nuovi Dati")
            new_x_values = []
            for i in range(min(2, n_features)):
                new_x_values.append(st.number_input(f"Inserisci valore per Feature {i+1}:", value=float(X[:, i].mean()), format="%.2f", key=f"new_x{i+1}_bc_val"))
            
            if st.button("Predici con Bagging Classifier", key="pred_bc_btn"):
                new_data_bc = np.array([new_x_values])
                pred_class_bc = model_bag_clf.predict(new_data_bc)
                pred_proba_bc = model_bag_clf.predict_proba(new_data_bc)
                st.success(f"Per i valori inseriti:")
                st.write(f"  - Classe Predetta: {pred_class_bc[0]}")
                st.write(f"  - Probabilità (Classe 0): {pred_proba_bc[0][0]:.4f}")
                st.write(f"  - Probabilità (Classe 1): {pred_proba_bc[0][1]:.4f}")
                
            # Salva le metriche di performance
            if "model_performances" not in st.session_state:
                st.session_state.model_performances = {}
                
            st.session_state.model_performances["Bagging (Classificazione)"] = calcola_metriche_classificazione(
                y_test, predictions_bc, y_pred_proba_bc
            )

        elif bagging_task_type == "Regressione" and task_type == "regression":
            st.subheader("Bagging per Regressione")
            
            # Utilizziamo solo la prima feature per visualizzazione semplice
            X_bag_train = X_train[:, 0].reshape(-1, 1)
            X_bag_test = X_test[:, 0].reshape(-1, 1)

            base_reg = DecisionTreeRegressor(max_depth=base_estimator_max_depth, random_state=42)
            model_bag_reg = BaggingRegressor(estimator=base_reg, n_estimators=n_estimators_bagging, random_state=42)
            model_bag_reg.fit(X_bag_train, y_train)
            predictions_br = model_bag_reg.predict(X_bag_test)
            mse_br = mean_squared_error(y_test, predictions_br)
            r2_br = r2_score(y_test, predictions_br)

            st.write(f"**Mean Squared Error (MSE) (Bagging Regressor):** {mse_br:.4f}")
            st.write(f"**R² Score (Bagging Regressor):** {r2_br:.4f}")

            st.markdown("#### Visualizzazione della Funzione Appresa (Bagging Regressione)")
            fig_br_func = go.Figure()
            fig_br_func.add_trace(go.Scatter(x=X_bag_test[:,0], y=y_test, mode='markers', name='Dati di Test'))
            x_range_br = np.sort(X_bag_test[:, 0]) 
            y_range_br = model_bag_reg.predict(x_range_br.reshape(-1, 1))
            fig_br_func.add_trace(go.Scatter(x=x_range_br, y=y_range_br, mode='lines', name='Predizione Bagging'))
            fig_br_func.update_layout(title="Bagging Regressione: Funzione Appresa", xaxis_title="Feature", yaxis_title="Target")
            st.plotly_chart(fig_br_func)

            st.markdown("#### Testa il Bagging Regressor con un Nuovo Dato")
            new_x_br = st.number_input("Valore Feature:", value=float(X[:, 0].mean()), format="%.2f", key="new_x_br")
            if st.button("Predici con Bagging Regressor", key="pred_br_btn"):
                predicted_y_br = model_bag_reg.predict(np.array([[new_x_br]]))
                st.success(f"Per Feature = {new_x_br:.2f}, il valore predetto è: {predicted_y_br[0]:.4f}")
                
            # Salva le metriche di performance
            if "model_performances" not in st.session_state:
                st.session_state.model_performances = {}
                
            st.session_state.model_performances["Bagging (Regressione)"] = {
                "mse": mse_br,
                "r2": r2_br
            }
        else:
            st.error(f"Incompatibilità tra il tipo di task del dataset ({task_type}) e il tipo di Bagging selezionato ({bagging_task_type}). Crea un nuovo dataset appropriato o seleziona un tipo di task compatibile.")

# --- Ensemble Methods: Boosting (AdaBoost) ---
elif selected_model == "Ensemble Methods: Boosting":
    st.header("Ensemble Methods: Boosting (AdaBoost)")
    st.markdown("""
    Il **Boosting** è una famiglia di algoritmi di ensemble learning che convertono una collezione di "weak learners" (modelli che performano solo leggermente meglio del caso) in uno "strong learner" (un modello con alta accuratezza).
    A differenza del Bagging (che addestra modelli in parallelo), il Boosting è un processo **sequenziale**.

    **AdaBoost (Adaptive Boosting)** è uno degli algoritmi di boosting più popolari:
    1.  Inizialmente, tutti i campioni di training hanno lo stesso peso.
    2.  Viene addestrato un weak learner (spesso un albero decisionale molto semplice, chiamato "stump" se ha `max_depth=1`).
    3.  I pesi dei campioni di training vengono aggiornati: i campioni classificati erroneamente dal modello corrente ricevono un peso maggiore, mentre quelli classificati correttamente ricevono un peso minore.
    4.  Un nuovo weak learner viene addestrato sul dataset pesato, concentrandosi così sugli errori del predecessore.
    5.  Questo processo viene ripetuto per un numero specificato di stimatori (`n_estimators`).
    6.  Le predizioni finali sono una combinazione pesata delle predizioni di tutti i weak learners (i modelli che hanno performato meglio sui dati pesati hanno un peso maggiore nella decisione finale).

    Useremo `n_estimators=50` (default). Per la classificazione, l'estimator base di default è `DecisionTreeClassifier(max_depth=1)`. Per la regressione, è `DecisionTreeRegressor(max_depth=3)`.
    """)

    if "dataset" not in st.session_state:
        st.warning("Devi prima generare un dataset! Vai alla sezione 'Dataset Generator'.")
    else:
        # Estrai il dataset dalla session_state
        X = st.session_state.dataset["X"]
        y = st.session_state.dataset["y"]
        X_train = st.session_state.dataset["X_train"]
        X_test = st.session_state.dataset["X_test"]
        y_train = st.session_state.dataset["y_train"]
        y_test = st.session_state.dataset["y_test"]
        task_type = st.session_state.dataset["task_type"]
        n_features = st.session_state.dataset["n_features"]

        boosting_task_type = st.selectbox("Scegli il tipo di task per AdaBoost:", ["Classificazione", "Regressione"], key="boosting_task_simplified")
        
        n_estimators_boosting = 50 # Default di AdaBoost

        st.info(f"Configurazione AdaBoost: n_estimators={n_estimators_boosting}. L'estimator base è scelto di default da scikit-learn (Decision Tree Stumps per classificazione, alberi con max_depth=3 per regressione).")

        if boosting_task_type == "Classificazione" and task_type == "classification":
            st.subheader("AdaBoost per Classificazione")
            
            # Utilizziamo max due features per visualizzazione
            X_boost_train = X_train[:, :min(2, n_features)]
            X_boost_test = X_test[:, :min(2, n_features)]

            # Per AdaBoostClassifier, l'estimator di default è DecisionTreeClassifier(max_depth=1)
            model_ada_clf = AdaBoostClassifier(n_estimators=n_estimators_boosting, random_state=42)
            model_ada_clf.fit(X_boost_train, y_train)
            predictions_ac = model_ada_clf.predict(X_boost_test)
            y_pred_proba_ac = model_ada_clf.predict_proba(X_boost_test)[:, 1]

            st.write(f"**Accuratezza (AdaBoost Classifier):** {accuracy_score(y_test, predictions_ac):.4f}")

            if n_features >= 2:
                st.markdown("#### Visualizzazione della Decision Boundary (AdaBoost Classificazione)")
                df_test_ac = pd.DataFrame(X_boost_test, columns=['Feature 1', 'Feature 2'])
                df_test_ac['Target'] = y_test
                fig_ac_boundary = px.scatter(df_test_ac, x='Feature 1', y='Feature 2', color='Target',
                                          color_discrete_map={0: 'blue', 1: 'red'},
                                          title=f"AdaBoost Classificazione: Decision Boundary")
                
                x_min_ac, x_max_ac = X[:, 0].min() - .5, X[:, 0].max() + .5
                y_min_ac, y_max_ac = X[:, 1].min() - .5, X[:, 1].max() + .5
                h_ac = .02 
                xx_ac, yy_ac = np.meshgrid(np.arange(x_min_ac, x_max_ac, h_ac), np.arange(y_min_ac, y_max_ac, h_ac))
                Z_ac = model_ada_clf.predict(np.c_[xx_ac.ravel(), yy_ac.ravel()])
                Z_ac = Z_ac.reshape(xx_ac.shape)
                contour_colors_ac = [[0, 'lightblue'], [1, 'lightcoral']]
                fig_ac_boundary.add_trace(go.Contour(x=xx_ac[0], y=yy_ac[:,0], z=Z_ac, showscale=False, colorscale=contour_colors_ac, opacity=0.4, hoverinfo='skip'))
                st.plotly_chart(fig_ac_boundary)

            st.markdown("#### Testa AdaBoost Classifier con Nuovi Dati")
            new_x_values = []
            for i in range(min(2, n_features)):
                new_x_values.append(st.number_input(f"Inserisci valore per Feature {i+1}:", value=float(X[:, i].mean()), format="%.2f", key=f"new_x{i+1}_ac_val"))
            
            if st.button("Predici con AdaBoost Classifier", key="pred_ac_btn"):
                new_data_ac = np.array([new_x_values])
                pred_class_ac = model_ada_clf.predict(new_data_ac)
                pred_proba_ac = model_ada_clf.predict_proba(new_data_ac)
                st.success(f"Per i valori inseriti:")
                st.write(f"  - Classe Predetta: {pred_class_ac[0]}")
                st.write(f"  - Probabilità (Classe 0): {pred_proba_ac[0][0]:.4f}")
                st.write(f"  - Probabilità (Classe 1): {pred_proba_ac[0][1]:.4f}")
                
            # Salva le metriche di performance
            if "model_performances" not in st.session_state:
                st.session_state.model_performances = {}
                
            st.session_state.model_performances["AdaBoost (Classificazione)"] = calcola_metriche_classificazione(
                y_test, predictions_ac, y_pred_proba_ac
            )

        elif boosting_task_type == "Regressione" and task_type == "regression":
            st.subheader("AdaBoost per Regressione")
            
            # Utilizziamo solo la prima feature per visualizzazione semplice
            X_boost_train = X_train[:, 0].reshape(-1, 1)
            X_boost_test = X_test[:, 0].reshape(-1, 1)

            # Per AdaBoostRegressor, l'estimator di default è DecisionTreeRegressor(max_depth=3)
            model_ada_reg = AdaBoostRegressor(n_estimators=n_estimators_boosting, random_state=42)
            model_ada_reg.fit(X_boost_train, y_train)
            predictions_ar = model_ada_reg.predict(X_boost_test)
            mse_ar = mean_squared_error(y_test, predictions_ar)
            r2_ar = r2_score(y_test, predictions_ar)

            st.write(f"**Mean Squared Error (MSE) (AdaBoost Regressor):** {mse_ar:.4f}")
            st.write(f"**R² Score (AdaBoost Regressor):** {r2_ar:.4f}")

            st.markdown("#### Visualizzazione della Funzione Appresa (AdaBoost Regressione)")
            fig_ar_func = go.Figure()
            fig_ar_func.add_trace(go.Scatter(x=X_boost_test[:,0], y=y_test, mode='markers', name='Dati di Test'))
            x_range_ar = np.sort(X_boost_test[:, 0]) 
            y_range_ar = model_ada_reg.predict(x_range_ar.reshape(-1, 1))
            fig_ar_func.add_trace(go.Scatter(x=x_range_ar, y=y_range_ar, mode='lines', name='Predizione AdaBoost'))
            fig_ar_func.update_layout(title="AdaBoost Regressione: Funzione Appresa", xaxis_title="Feature", yaxis_title="Target")
            st.plotly_chart(fig_ar_func)

            st.markdown("#### Testa AdaBoost Regressor con un Nuovo Dato")
            new_x_ar = st.number_input("Valore Feature:", value=float(X[:, 0].mean()), format="%.2f", key="new_x_ar")
            if st.button("Predici con AdaBoost Regressor", key="pred_ar_btn"):
                predicted_y_ar = model_ada_reg.predict(np.array([[new_x_ar]]))
                st.success(f"Per Feature = {new_x_ar:.2f}, il valore predetto è: {predicted_y_ar[0]:.4f}")
                
            # Salva le metriche di performance
            if "model_performances" not in st.session_state:
                st.session_state.model_performances = {}
                
            st.session_state.model_performances["AdaBoost (Regressione)"] = {
                "mse": mse_ar,
                "r2": r2_ar
            }
        else:
            st.error(f"Incompatibilità tra il tipo di task del dataset ({task_type}) e il tipo di AdaBoost selezionato ({boosting_task_type}). Crea un nuovo dataset appropriato o seleziona un tipo di task compatibile.")

# --- Naive Bayes ---
elif selected_model == "Naive Bayes":
    st.header("Naive Bayes")
    st.markdown("""
    Naive Bayes è una famiglia di classificatori probabilistici basati sul Teorema di Bayes con un'assunzione forte (ma "ingenua"/"naive"): 
    **indipendenza condizionale tra le features dato il valore della classe target**.
    
    **Teorema di Bayes**: P(y|X) = P(X|y) * P(y) / P(X)
    
    Dove:
    - P(y|X) è la probabilità a posteriori della classe y dato il vettore di feature X
    - P(X|y) è la verosimiglianza delle features dato y
    - P(y) è la probabilità a priori della classe y
    - P(X) è la probabilità a priori delle features (costante per tutte le classi)
    
    In questa app, usiamo **Gaussian Naive Bayes**, che assume che le features seguano una distribuzione gaussiana (normale) all'interno di ogni classe.
    
    **Vantaggi di Naive Bayes**:
    - Semplice e veloce
    - Funziona bene con dataset di piccole dimensioni
    - Efficace con dati ad alta dimensionalità
    - Richiede poca capacità computazionale
    - Può essere usato per problemi multi-classe
    
    **Limitazioni**:
    - L'assunzione di indipendenza condizionale è spesso violata nella pratica
    - Per questo è chiamato "naive" (ingenuo)
    - Altre varianti di Naive Bayes includono Multinomial NB (per dati discreti come conteggi), Bernoulli NB (per dati binari) e Complement NB.
    """)

    if "dataset" not in st.session_state:
        st.warning("Devi prima generare un dataset! Vai alla sezione 'Dataset Generator'.")
    elif st.session_state.dataset["task_type"] != "classification":
        st.error("Naive Bayes è un algoritmo di classificazione. Crea un dataset di classificazione per usare questo modello.")
    else:
        # Estrai il dataset dalla session_state (solo per classificazione)
        X = st.session_state.dataset["X"]
        y = st.session_state.dataset["y"]
        X_train = st.session_state.dataset["X_train"]
        X_test = st.session_state.dataset["X_test"]
        y_train = st.session_state.dataset["y_train"]
        y_test = st.session_state.dataset["y_test"]
        n_features = st.session_state.dataset["n_features"]
        
        st.subheader("Gaussian Naive Bayes per Classificazione")
        
        # Utilizziamo max due features per visualizzazione
        X_nb_train = X_train[:, :min(2, n_features)]
        X_nb_test = X_test[:, :min(2, n_features)]

        model_nb = GaussianNB()
        model_nb.fit(X_nb_train, y_train)
        predictions_nb = model_nb.predict(X_nb_test)
        y_pred_proba_nb = model_nb.predict_proba(X_nb_test)[:, 1]

        st.write(f"**Accuratezza (Gaussian Naive Bayes):** {accuracy_score(y_test, predictions_nb):.4f}")
        
        # Informazioni aggiuntive per comprendere il modello
        st.write("#### Parametri del Modello:")
        st.write("**Probabilità a priori delle classi (P(y)):**")
        for i, prior in enumerate(model_nb.class_prior_):
            st.write(f"  - Classe {i}: {prior:.4f}")
        
        st.write("**Media delle features per classe (parametro μ della distribuzione Gaussiana):**")
        for i, mean in enumerate(model_nb.theta_):
            st.write(f"  - Classe {i}: {', '.join([f'Feature {j+1}: {v:.4f}' for j, v in enumerate(mean[:min(2, n_features)])])}")
        
        st.write("**Varianza delle features per classe (parametro σ² della distribuzione Gaussiana):**")
        for i, var in enumerate(model_nb.var_):
            st.write(f"  - Classe {i}: {', '.join([f'Feature {j+1}: {v:.4f}' for j, v in enumerate(var[:min(2, n_features)])])}")
        
        st.write("*Nota: Durante la predizione, NB calcola la probabilità di ogni classe dato il nuovo esempio, usando la distribuzione normale con questi parametri, e sceglie la classe con la probabilità più alta.*")

        if n_features >= 2:
            st.markdown("#### Visualizzazione della Decision Boundary (Naive Bayes)")
            df_test_nb = pd.DataFrame(X_nb_test, columns=['Feature 1', 'Feature 2'])
            df_test_nb['Target'] = y_test
            fig_nb_boundary = px.scatter(df_test_nb, x='Feature 1', y='Feature 2', color='Target',
                                      color_discrete_map={0: 'blue', 1: 'red'},
                                      title="Gaussian Naive Bayes: Decision Boundary")
            
            x_min_nb, x_max_nb = X[:, 0].min() - .5, X[:, 0].max() + .5
            y_min_nb, y_max_nb = X[:, 1].min() - .5, X[:, 1].max() + .5
            h_nb = .02
            xx_nb, yy_nb = np.meshgrid(np.arange(x_min_nb, x_max_nb, h_nb), np.arange(y_min_nb, y_max_nb, h_nb))
            Z_nb = model_nb.predict(np.c_[xx_nb.ravel(), yy_nb.ravel()])
            Z_nb = Z_nb.reshape(xx_nb.shape)
            contour_colors_nb = [[0, 'lightblue'], [1, 'lightcoral']]
            fig_nb_boundary.add_trace(go.Contour(x=xx_nb[0], y=yy_nb[:,0], z=Z_nb, showscale=False, colorscale=contour_colors_nb, opacity=0.4, hoverinfo='skip'))
            st.plotly_chart(fig_nb_boundary)

        st.markdown("#### Testa Naive Bayes con Nuovi Dati")
        new_x_values = []
        for i in range(min(2, n_features)):
            new_x_values.append(st.number_input(f"Inserisci valore per Feature {i+1}:", value=float(X[:, i].mean()), format="%.2f", key=f"new_x{i+1}_nb_val"))
        
        if st.button("Predici con Naive Bayes", key="pred_nb_btn"):
            new_data_nb = np.array([new_x_values])
            pred_class_nb = model_nb.predict(new_data_nb)
            pred_proba_nb = model_nb.predict_proba(new_data_nb)
            st.success(f"Per i valori inseriti:")
            st.write(f"  - Classe Predetta: {pred_class_nb[0]}")
            st.write(f"  - Probabilità (Classe 0): {pred_proba_nb[0][0]:.4f}")
            st.write(f"  - Probabilità (Classe 1): {pred_proba_nb[0][1]:.4f}")
            
        # Salva le metriche di performance
        if "model_performances" not in st.session_state:
            st.session_state.model_performances = {}
            
        st.session_state.model_performances["Naive Bayes"] = calcola_metriche_classificazione(
            y_test, predictions_nb, y_pred_proba_nb
        )
