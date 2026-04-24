
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

if not os.path.exists('plots'):
    os.makedirs('plots')
CSV_PATH = "car_price_dataset.csv"  
os.makedirs("plots", exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")

df = pd.read_csv(CSV_PATH)

print("=" * 55)
print("   ANALYSE EXPLORATOIRE — CAR PRICE DATASET")
print("=" * 55)

print(f"\n Dimensions       : {df.shape[0]} lignes × {df.shape[1]} colonnes")
print(f"\n Colonnes         : {df.columns.tolist()}")

print("\n Aperçu des 5 premières lignes :")
print(df.head().to_string(index=False))

print("\n  Valeurs manquantes :")
missing = df.isnull().sum()
arrow = "->"
if missing.sum() == 0:
    print("   Aucune valeur manquante ")
else:
    print(missing[missing > 0])

print("\nStatistiques des variables numériques :")
print(df.describe().round(2).to_string())

print("\n Variables catégorielles :")
for col in ["Brand", "Fuel_Type", "Transmission"]:
    print(f"   {col} {arrow}  {df[col].nunique()} valeurs : {df[col].unique().tolist()}")

plt.figure(figsize=(9, 4))
sns.histplot(df["Price"], bins=40, kde=True, color="#2E86AB")
plt.title("Distribution du Prix des Voitures", fontsize=14, fontweight="bold")
plt.xlabel("Prix ($)")
plt.ylabel("Fréquence")
plt.tight_layout()
plt.savefig("plots/01_distribution_prix.png", dpi=150)
plt.show()
print("\n Plot 1 sauvegardé : plots/01_distribution_prix.png")

plt.figure(figsize=(10, 5))
order = df.groupby("Brand")["Price"].median().sort_values(ascending=False).index
sns.boxplot(x="Brand", y="Price", data=df, order=order, palette="Set2")
plt.title("Prix selon la Marque", fontsize=14, fontweight="bold")
plt.xlabel("Marque")
plt.ylabel("Prix ($)")
plt.tight_layout()
plt.savefig("plots/02_prix_par_marque.png", dpi=150)
plt.show()
print("Plot 2 sauvegardé : plots/02_prix_par_marque.png")

plt.figure(figsize=(8, 4))
sns.boxplot(x="Fuel_Type", y="Price", data=df, palette="pastel")
plt.title("Prix selon le Type de Carburant", fontsize=14, fontweight="bold")
plt.xlabel("Type de Carburant")
plt.ylabel("Prix ($)")
plt.tight_layout()
plt.savefig("plots/03_prix_par_carburant.png", dpi=150)
plt.show()
print(" Plot 3 sauvegardé : plots/03_prix_par_carburant.png")

plt.figure(figsize=(7, 4))
sns.boxplot(x="Transmission", y="Price", data=df, palette="Set3")
plt.title("Prix selon la Transmission", fontsize=14, fontweight="bold")
plt.xlabel("Transmission")
plt.ylabel("Prix ($)")
plt.tight_layout()
plt.savefig("plots/04_prix_par_transmission.png", dpi=150)
plt.show()
print(" Plot 4 sauvegardé : plots/04_prix_par_transmission.png")

plt.figure(figsize=(9, 4))
sns.scatterplot(x="Mileage", y="Price", hue="Fuel_Type", data=df, alpha=0.6)
plt.title("Kilométrage vs Prix", fontsize=14, fontweight="bold")
plt.xlabel("Kilométrage")
plt.ylabel("Prix ($)")
plt.tight_layout()
plt.savefig("plots/05_mileage_vs_prix.png", dpi=150)
plt.show()
print(" Plot 5 sauvegardé : plots/05_mileage_vs_prix.png")

plt.figure(figsize=(9, 4))
sns.scatterplot(x="Horsepower", y="Price", hue="Brand", data=df, alpha=0.7)
plt.title("Puissance Moteur vs Prix", fontsize=14, fontweight="bold")
plt.xlabel("Puissance (CV)")
plt.ylabel("Prix ($)")
plt.tight_layout()
plt.savefig("plots/06_horsepower_vs_prix.png", dpi=150)
plt.show()
print(" Plot 6 sauvegardé : plots/06_horsepower_vs_prix.png")

plt.figure(figsize=(12, 4))
avg_price_year = df.groupby("Model_Year")["Price"].mean().reset_index()
sns.lineplot(x="Model_Year", y="Price", data=avg_price_year, marker="o", color="#E84855")
plt.title("Évolution du Prix Moyen par Année", fontsize=14, fontweight="bold")
plt.xlabel("Année du Modèle")
plt.ylabel("Prix Moyen ($)")
plt.tight_layout()
plt.savefig("plots/07_prix_par_annee.png", dpi=150)
plt.show()
print("Plot 7 sauvegardé : plots/07_prix_par_annee.png")

numeric_cols = ["Model_Year", "Engine_Size", "Mileage", "Doors",
                "Owner_Count", "Horsepower", "Price"]
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(9, 7))
mask = pd.DataFrame(False, index=corr_matrix.index, columns=corr_matrix.columns)
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, mask=mask.values)
plt.title("Matrice de Corrélation", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/08_correlation_matrix.png", dpi=150)
plt.show()
print(" Plot 8 sauvegardé : plots/08_correlation_matrix.png")

print("\n Corrélation de chaque variable avec le Prix :")
corr_with_price = corr_matrix["Price"].drop("Price").sort_values(ascending=False)
for var, val in corr_with_price.items():
    bar = "#" * int(abs(val) * 20)
    sign = "+" if val > 0 else "-"
    print(f"   {var:<15} {sign}{abs(val):.3f}  {bar}")

print("\n" + "=" * 55)
print("ANALYSE TERMINÉE — 8 graphiques dans /plots")
print("=" * 55)

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os


df = pd.read_csv("car_price_dataset.csv")

print("=" * 55)
print("   NETTOYAGE & PRÉPARATION — CAR PRICE DATASET")
print("=" * 55)

print(f"\n Dimensions initiales : {df.shape[0]} lignes × {df.shape[1]} colonnes")


df = df.drop(columns=["Car_ID"])

print("\n Colonne 'Car_ID' supprimée (inutile pour la prédiction)")


print("\n Vérification des valeurs manquantes :")
missing = df.isnull().sum()

if missing.sum() == 0:
    print("   Aucune valeur manquante ")
else:
    print(missing[missing > 0])

    num_cols = df.select_dtypes(include="number").columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"   {col} → valeurs manquantes remplacées par la médiane ({median_val})")

    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"   {col} → valeurs manquantes remplacées par le mode ('{mode_val}')")


nb_doublons = df.duplicated().sum()
print(f"\nDoublons détectés : {nb_doublons}")

if nb_doublons > 0:
    df = df.drop_duplicates()
    print(f"    {nb_doublons} doublon(s) supprimé(s)")
else:
    print("   Aucun doublon ")


print("\n🔎 Détection des valeurs aberrantes (méthode IQR) :")

cols_outliers = ["Price", "Mileage", "Horsepower", "Engine_Size"]

for col in cols_outliers:
    Q1 = df[col].quantile(0.25)   # 
    Q3 = df[col].quantile(0.75)   
    IQR = Q3 - Q1                

    borne_basse = Q1 - 1.5 * IQR
    borne_haute = Q3 + 1.5 * IQR

    nb_outliers = df[(df[col] < borne_basse) | (df[col] > borne_haute)].shape[0]
    print(f"   {col:<15} → {nb_outliers} valeur(s) aberrante(s)  [bornes : {borne_basse:.1f} — {borne_haute:.1f}]")

    df = df[(df[col] >= borne_basse) & (df[col] <= borne_haute)]

print(f"\n Dimensions après nettoyage : {df.shape[0]} lignes × {df.shape[1]} colonnes")


print("\nEncodage des variables catégorielles :")

le = LabelEncoder()

cat_cols = ["Brand", "Fuel_Type", "Transmission"]

for col in cat_cols:
    df[col] = le.fit_transform(df[col])
    print(f"   {col} → converti en nombres (0, 1, 2...)")

print("\n   Aperçu après encodage :")
print(df[cat_cols].head(3).to_string(index=False))


X = df.drop(columns=["Price"])
y = df["Price"]

print(f"\n Séparation X / y :")
print(f"   X (features) : {X.shape[1]} colonnes → {X.columns.tolist()}")
print(f"   y (cible)    : colonne 'Price'")


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print("\n Normalisation des features (StandardScaler) :")
print("   Avant normalisation — colonne 'Mileage' :")
print(f"   min={X['Mileage'].min():.0f}, max={X['Mileage'].max():.0f}")
print("   Après normalisation — colonne 'Mileage' :")
print(f"   min={X_scaled['Mileage'].min():.3f}, max={X_scaled['Mileage'].max():.3f}")


os.makedirs("data_clean", exist_ok=True)

X_scaled.to_csv("data_clean/X_clean.csv", index=False)
y.to_csv("data_clean/y_clean.csv", index=False)

print("\n Données sauvegardées dans /data_clean :")
print("    X_clean.csv  (features normalisées)")
print("    y_clean.csv  (prix cibles)")

print("\n" + "=" * 55)
print(" NETTOYAGE TERMINÉ — Données prêtes pour la modélisation !")
print("=" * 55)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os


X = pd.read_csv("data_clean/X_clean.csv")
y = pd.read_csv("data_clean/y_clean.csv").squeeze()  

os.makedirs("plots", exist_ok=True)

print("=" * 60)
print("   MODÉLISATION — PRÉDICTION DU PRIX DES VOITURES")
print("=" * 60)
print(f"\n X : {X.shape[0]} lignes × {X.shape[1]} colonnes")
print(f" y : {y.shape[0]} valeurs de prix")

# ════════════════════

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       
    random_state=42      
)

print(f"\n  Division Train / Test :")
print(f"   Entraînement : {X_train.shape[0]} voitures (80%)")
print(f"   Test         : {X_test.shape[0]} voitures (20%)")


modeles = {
    "Régression Linéaire": LinearRegression(),
    "Arbre de Décision"  : DecisionTreeRegressor(random_state=42),
    "Random Forest"      : RandomForestRegressor(n_estimators=100, random_state=42)
}

print("\n" + "=" * 60)
print("   RÉSULTATS DES MODÈLES")
print("=" * 60)

resultats = {} 

for nom, modele in modeles.items():

    modele.fit(X_train, y_train)

    y_pred = modele.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    resultats[nom] = {"MAE": mae, "RMSE": rmse, "R2": r2, "y_pred": y_pred}

    print(f"\n {nom}")
    print(f"   MAE  (Erreur Absolue Moyenne)  : {mae:,.2f} $")
    print(f"   RMSE (Racine Erreur Quadratique): {rmse:,.2f} $")
    print(f"   R²   (Score d'explication)      : {r2:.4f}  ({r2*100:.1f}%)")


meilleur = max(resultats, key=lambda x: resultats[x]["R2"])
print("\n" + "=" * 60)
print(f" Meilleur modèle : {meilleur}")
print(f"   R² = {resultats[meilleur]['R2']*100:.1f}% — le modèle explique")
print(f"   {resultats[meilleur]['R2']*100:.1f}% de la variation des prix.")
print("=" * 60)


noms    = list(resultats.keys())
r2_vals = [resultats[n]["R2"] for n in noms]
couleurs = ["#E84855" if n == meilleur else "#2E86AB" for n in noms]

plt.figure(figsize=(8, 4))
bars = plt.bar(noms, r2_vals, color=couleurs, edgecolor="white", linewidth=1.5)
plt.ylim(0, 1)
plt.title("Comparaison des modèles — Score R²", fontsize=14, fontweight="bold")
plt.ylabel("R² (plus proche de 1 = meilleur)")


for bar, val in zip(bars, r2_vals):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{val:.3f}", ha="center", va="bottom", fontweight="bold")

plt.tight_layout()
plt.savefig("plots/09_comparaison_modeles.png", dpi=150)
plt.show()
print("\n Plot 9 sauvegardé : plots/09_comparaison_modeles.png")


y_pred_meilleur = resultats[meilleur]["y_pred"]

plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred_meilleur, alpha=0.5, color="#2E86AB", edgecolors="white", linewidth=0.5)

# Ligne idéale : si le modèle était parfait, tous les points seraient sur cette ligne
min_val = min(y_test.min(), y_pred_meilleur.min())
max_val = max(y_test.max(), y_pred_meilleur.max())
plt.plot([min_val, max_val], [min_val, max_val], color="#E84855", linestyle="--", linewidth=2, label="Prédiction parfaite")

plt.title(f"Prix Réels vs Prix Prédits\n({meilleur})", fontsize=13, fontweight="bold")
plt.xlabel("Prix Réels ($)")
plt.ylabel("Prix Prédits ($)")
plt.legend()
plt.tight_layout()
plt.savefig("plots/10_reel_vs_predit.png", dpi=150)
plt.show()
print("✅ Plot 10 sauvegardé : plots/10_reel_vs_predit.png")


rf_model = modeles["Random Forest"]
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=True)

plt.figure(figsize=(8, 5))
importances.plot(kind="barh", color="#2E86AB", edgecolor="white")
plt.title("Importance des Variables (Random Forest)", fontsize=13, fontweight="bold")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("plots/11_importance_features.png", dpi=150)
plt.show()
print(" Plot 11 sauvegardé : plots/11_importance_features.png")


print("\n" + "=" * 60)
print("   EXEMPLE DE PRÉDICTION")
print("=" * 60)

exemple = X_test.iloc[[0]]
prix_reel   = y_test.iloc[0]
prix_predit = modeles[meilleur].predict(exemple)[0]

print(f"\n Voiture de test :")
print(exemple.to_string(index=False))
print(f"\n    Prix réel    : {prix_reel:,.2f} $")
print(f"    Prix prédit  : {prix_predit:,.2f} $")
print(f"    Écart        : {abs(prix_reel - prix_predit):,.2f} $")

print("\n" + "=" * 60)
print(" MODÉLISATION TERMINÉE — 3 nouveaux graphiques dans /plots")
print("=" * 60)
