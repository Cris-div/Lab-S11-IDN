from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def main():
    # Columnas: Edad, Ingresos_Mensuales, Deuda_Mensual.
    X = [
        [22, 1200, 800], [25, 1500, 700], [30, 2500, 600],
        [35, 3000, 900], [40, 3500, 1200], [45, 4000, 1800],
        [23, 1000, 850], [28, 1800, 1100], [32, 2700, 700],
        [38, 3200, 1500], [50, 4500, 2200], [26, 1400, 1000],
    ]
    y = [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    predicciones = modelo.predict(X_test)

    print("Riesgo: 0 = bajo, 1 = alto")
    print("Valores reales:", y_test)
    print("Predicciones:", predicciones.tolist())
    print(f"Accuracy: {accuracy_score(y_test, predicciones):.4f}")


if __name__ == "__main__":
    main()
