from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# données et étiquettes
y_true = ["femme"] * 100 + ["homme"] * 100  # 11 femmes réelles, 8 hommes réels
y_pred = ["femme"] * 30 + ["homme"] * 70 + ["homme"] * 50 + ["femme"] * 50  # prédiction

# Calcul de la matrice de confusion
conf_matrix = confusion_matrix(y_true, y_pred, labels=["femme", "homme"])

ax = sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Prédit Femme", "Prédit Homme"], yticklabels=["Images\n Générées Femmes", "Images\n Générées Hommes"])
plt.ylabel('Valeurs Images Générées')
plt.xlabel('Valeurs Prédites')
plt.show()

