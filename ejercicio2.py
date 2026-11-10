from math import sqrt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def main():
    # Columnas: Personas, Electrodomesticos, Horas_Uso_Diario.
    X = [
        [1, 4, 3], [2, 5, 4], [3, 7, 5], [4, 8, 6],
        [5, 10, 7], [2, 6, 5], [3, 8, 6], [4, 9, 7],
        [5, 11, 8], [6, 12, 9], [1, 3, 2], [3, 6, 4],
    ]
    y = [80, 120, 180, 240, 310, 150, 220, 280, 350, 420, 60, 160]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    predicciones = modelo.predict(X_test)
    mse = mean_squared_error(y_test, predicciones)
    rmse = sqrt(mse)

    print("Consumo real (kWh):", y_test)
    print("Consumo predicho (kWh):", predicciones.round(2).tolist())
    print(f"MSE: {mse:.2f} kWh²")
    print(f"RMSE: {rmse:.2f} kWh")


if __name__ == "__main__":
    main()
