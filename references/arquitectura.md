# Arquitectura — MiBici (Equipo 16)

Arquitectura local y por lotes: los scripts del repositorio llevan los datos del portal a MLflow, y
`mibici/config.py` fija qué datos, fechas y semillas usa cada run.

```mermaid
flowchart TB
    portal["Portal MiBici<br/>24 CSV mensuales + catálogo de estaciones"]

    subgraph repo["Repositorio en GitHub · main protegida"]
        config["mibici/config.py<br/>rutas, fechas de corte y semillas"]
        dataset["mibici/dataset.py<br/>descarga, normaliza, quita datos personales"]
        manifiesto["references/manifiesto_datos.json<br/>filas y SHA-256 por archivo"]
        features["mibici/features.py<br/>objetivos P1-P3, variables, partición temporal"]
        train["mibici/modeling/train.py<br/>baselines y candidatos (scikit-learn)"]
        tracking["mibici/tracking.py<br/>tags equipo, pregunta y commit"]
    end

    parquet[("data/processed<br/>Parquet local, fuera de Git")]
    mlflow[("MLflow · mibici-equipo16<br/>params, métricas, artefactos, tags")]

    portal --> dataset
    config --> dataset & features & train
    dataset --> parquet
    dataset --> manifiesto
    parquet --> features --> train
    train --> tracking --> mlflow
    manifiesto -- versión de datos --> tracking
```

| Herramienta | Uso | Por qué |
|---|---|---|
| Git y GitHub | Código, PRs y revisión con `main` protegida | Trazabilidad de cada cambio y revisión por pares (GitHub Flow) |
| Cookiecutter Data Science | Estructura del repositorio | Convención conocida: cada cosa tiene un lugar |
| pandas y Parquet (pyarrow) | Lectura de CSV y almacenamiento normalizado | Formato columnar con tipos y compresión; más rápido que CSV |
| Manifiesto con `hashlib` | Versión de los datos | Detecta cualquier cambio en los datos sin servidor adicional |
| scikit-learn | Baselines, modelos y métricas | Interfaz única para modelos, `Dummy*` y métricas de evaluación |
| MLflow Tracking | Runs con params, métricas, artefactos y tags | Compara runs y los liga a una versión de datos y un commit |
| typer, loguru, ruff, pytest | Terminal, registro, estilo y pruebas | Vienen con la plantilla; ejecución sin pasos manuales |
