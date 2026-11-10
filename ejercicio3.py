from sklearn.ensemble import RandomForestClassifier


def main():
    # Columnas: Visitas_Mes, Meses_Inscrito, Satisfaccion.
    X = [
        [2, 1, 3], [4, 2, 4], [8, 6, 7], [10, 8, 8],
        [12, 12, 9], [3, 1, 4], [5, 3, 5], [9, 7, 8],
        [11, 10, 9], [6, 4, 6], [1, 1, 2], [7, 5, 7],
    ]
    y = [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1]

    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X, y)

    nuevo_cliente = [[6, 5, 7]]
    prediccion = int(modelo.predict(nuevo_cliente)[0])
    print("Cliente: Visitas_Mes=6, Meses_Inscrito=5, Satisfaccion=7")
    print(f"Renueva: {prediccion}")
    print("Sí renueva su membresía." if prediccion == 1 else "No renueva su membresía.")


if __name__ == "__main__":
    main()
