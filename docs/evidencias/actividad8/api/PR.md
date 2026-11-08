## Descripción

Expone el modelo P1 de CH1 mediante FastAPI. `POST /predict` acepta los 11 campos
reales del contrato y devuelve una predicción finita positiva por registro;
entradas inválidas producen 422. Los tres rezagos mantienen su carácter opcional.

La app carga el modelo una vez en lifespan y verifica su firma. `/health` responde
503 si no se pudo cargar o la firma es incompatible; `/docs` sigue disponible.
Los errores de inferencia y las salidas inválidas producen 500 controlado, sin
filtrar rutas, URIs, tokens ni trazas internas. Solo se exponen cuatro metadatos
públicos, con `response_model` y conversión NumPy explícita.
Se rechazan booleanos también en salidas mixtas. Cada petición conserva los
metadatos del modelo usado, aunque otra petición lo marque como no disponible.

## Dependencia y alcance

Base: `main` en `f4eebfc`, que ya incluye el trabajo de CH1 mediante el PR #15.
Se reutilizan su loader y su contrato sin modificarlos. Este PR se dirige a main.
Archivos propios: `api.py`, `schemas.py`, `tests/test_api.py`,
`requirements-api.txt` y la evidencia de `docs/evidencias/actividad8/api/`.
No se modifican README, Containerfile, requisitos del modelo, loader ni su contrato.

## Validación

- **66 pruebas HTTP aprobadas** con TestClient, loader real y predictor simulado;
  sin omisiones, con 1 advertencia de deprecación de Starlette. Ver `pytest-api.txt`.
- **279 pruebas totales aprobadas**, 113 advertencias, sin fallos ni omisiones
  (450.31 s). Ruff y formato correctos; salidas en `pytest-suite.txt` y `ruff.txt`.
- Uvicorn real con el champion existente del Registry local (versión 1):
  `/health=200`, `/predict=200`, entrada incompleta `422`, `/docs=200`.
- Predicciones HTTP iguales a las del loader:
  `[40.697957479000046, 276.77983750628846]`.
- No se reentrenó ni se cambió un alias. CH1 documenta versión 2 en su Registry;
  la evidencia local declara su versión 1, sin confundir ambos entornos.

La reproducción, dependencias exactas, advertencias y JSON están en
`docs/evidencias/actividad8/api/VALIDACION.md` y `respuesta_local.json`.
La ejecución real dentro del contenedor queda para el PR de HH2.

Revisor previsto: Horacio (`HHSavenger`); solicitud de revisión y merge pendientes.
