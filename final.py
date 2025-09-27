import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
#  TP : Régression logistique "from scratch" (1 neurone)
#  Objectif : prédire si un animal a des PLUMES (1) ou des POILS (0)
#  Features : taille, poids, vole (0/1), pond_des_oeufs (0/1)
#  Perte : Binary Cross-Entropy (log loss)
#  Optimisation : descente de gradient pleine-batch + LR decay + early stopping
# ============================================================

# =============================
#  Chargement & préparation
# =============================
df = pd.read_csv("animaux_classification.csv")

# Colonnes d'entrée choisies (numériques déjà prêtes)
feature_cols = ["taille", "poids", "vole", "pond_des_oeufs"]
X_raw = df[feature_cols].values.astype(float)

# Colonne cible (1 = plume, 0 = poil)
y = df["classe"].values.reshape(-1, 1).astype(float)

#  Standardisation : indispensable pour stabiliser l'entraînement.
#    On met chaque feature à moyenne 0 et variance 1 pour que :
#    - un même learning rate convienne à toutes les dimensions,
#    - le paysage d'optimisation soit mieux conditionné.
X_mean = X_raw.mean(axis=0, keepdims=True)
X_std  = X_raw.std(axis=0, keepdims=True) + 1e-8  # +eps pour éviter la division par 0
X = (X_raw - X_mean) / X_std

# =============================
#  Neurone logistique
# =============================
def sigmoid(z):
    """σ(z) = 1 / (1 + e^(-z)).
    Interprétation : probabilité prédite P(y=1|x)."""
    return 1.0 / (1.0 + np.exp(-z))

def forward(X, w, b):
    """Propagation avant :
       z = X @ w + b   (combinaison linéaire)
       a = σ(z)        (probabilité plume)"""
    z = X @ w + b
    a = sigmoid(z)
    return z, a

def logloss(a, y):
    """Binary Cross-Entropy moyenne :
       L = - mean( y log a + (1-y) log(1-a) )
       (eps évite log(0))"""
    eps = 1e-12
    return - (y*np.log(a+eps) + (1-y)*np.log(1-a+eps)).mean()

def gradients(X, a, y):
    """Gradients analytiques (descente de gradient batch) :
       dL/dz = a - y
       dL/dw = X^T (a - y) / m
       dL/db = mean(a - y)"""
    dz = a - y
    dw = (X.T @ dz) / len(X)
    db = dz.mean()
    return dw, db

# =============================
#  Démo sur 1 exemple (optionnel)
#  Utile pour vérifier les formules (z, a, L, ∂L/∂w, ∂L/∂b)
# =============================
w = np.zeros((X.shape[1], 1))
b = 0.0
z1, a1 = forward(X[:1], w, b)
L1 = logloss(a1, y[:1])
dw1, db1 = gradients(X[:1], a1, y[:1])
print("=== Exemple unique ===")
print(f"x={X[0].round(3)}  y={int(y[0,0])}  z={z1.item():.6f}  a={a1.item():.6f}  L={L1:.6f}")
print(f"dL/dw={dw1.ravel()}  dL/db={db1:.6f}\n")

# =============================
#  Entraînement + LR decay + early stopping
# =============================
np.random.seed(0)
# Initialisation petite aléatoire (évite symétries triviales tout en restant stable)
w = np.random.randn(X.shape[1], 1) * 0.01
b = 0.0

lr0 = 1e-1          # learning rate initial
decay = 0.9992      # décroissance progressive: lr_epoch = lr0 * decay**epoch
epochs = 2500       # demandé par l’enseignant
patience = 400      # nb d'epochs sans amélioration avant arrêt anticipé

best_loss = float("inf")
best_w, best_b = None, None
stall = 0

loss_history = []   # on trace la courbe de log-loss d'entraînement

for epoch in range(1, epochs + 1):
    # Forward
    z, a = forward(X, w, b)
    L = logloss(a, y)
    loss_history.append(L)

    # Backprop (gradients fermés) + mise à jour GD
    dw, db = gradients(X, a, y)
    lr = lr0 * (decay ** epoch)
    w -= lr * dw
    b -= lr * db

    # Early stopping : on mémorise le meilleur modèle
    if L < best_loss - 1e-7:
        best_loss = L
        best_w, best_b = w.copy(), float(b)
        stall = 0
    else:
        stall += 1
        if stall >= patience:
            # on restaure les meilleurs poids et on arrête
            w, b = best_w, best_b
            print(f"[early stop @ {epoch}] best_loss={best_loss:.6f}")
            break

    # Log périodique (perte/accuracy, LR, poids)
    if epoch % 250 == 0 or epoch == 1:
        pred = (a > 0.5).astype(int)
        acc = (pred == y).mean()
        w_flat = ", ".join([f"{wi[0]:+.3f}" for wi in w])
        print(f"[{epoch:04d}] loss={L:.6f}  acc={acc:.3f}  lr={lr:.5f}  w=[{w_flat}]  b={b:+.3f}")


# Sauvegarde de l'historique de la perte (optionnel)
# Utile si on veut recharger les valeurs plus tard sans réentraîner.
# loss_history = np.array(loss_history, dtype=float)
# np.save("loss_history.npy", loss_history)

# =============================
#  Courbe log-loss
#  Interprétation : doit décroître et se stabiliser; si elle remonte,
#  c’est souvent signe d’overfitting (pas le cas ici en full-batch).
# =============================
plt.figure()
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Log loss (BCE)")
plt.title("Courbe d'entraînement (log loss)")
plt.grid(True)
plt.tight_layout()
plt.savefig("logloss_curve.png", dpi=150)

# =============================
#  Prédiction
#  Remarque : on réapplique EXACTEMENT la même standardisation
#  que celle utilisée au train (mêmes X_mean, X_std).
# =============================
def predict_animal(taille, poids, vole, pond_des_oeufs):
    x = np.array([[taille, poids, vole, pond_des_oeufs]], dtype=float)
    x = (x - X_mean) / X_std
    _, a = forward(x, w, b)
    proba_plume = a.item()           # proba scalaire P(y=1|x)
    label = int(proba_plume > 0.5)   # 1 = plume, 0 = poil
    return proba_plume, label

# =============================
#  Tests (sanity checks)
#  - Oiseau typique : vole=1, pond=1 => ≈ plume
#  - Mammifère typique : vole=0, pond=0 => ≈ poil
#  - Ornithorynque : vole=0, pond=1, petit/léger => cas d'exception
#    -> sans exemples d'exception dans le train, la régression logistique
#       tend à le classer "plume".
# =============================
p, lbl = predict_animal(1.1, 0.9, 1, 1)
print(f"\nExemple oiseau-like -> P(plume)={p:.3f}  label={lbl}")

p, lbl = predict_animal(1.7, 65, 0, 0)
print(f"Exemple mammifère-like -> P(plume)={p:.3f}  label={lbl}")

p, lbl = predict_animal(0.5, 2.0, 0, 1)
print(f"Exemple ornithorynque-like -> P(plume)={p:.3f}  label={lbl}")
