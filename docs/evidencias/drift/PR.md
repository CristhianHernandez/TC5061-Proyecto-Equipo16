## Descripción

PR #2 del Entregable 8 (LH3 → revisa HH2). Agrega el reporte de drift y calidad de datos con
Evidently para el champion de P1 (`mibici-equipo16-p1-demanda`, versión 2) sobre los conjuntos del
PR #19 de CH1: referencia (train oct-2024 a mar-2026) contra octubre de 2026.

- `mibici/monitoring/evidently_report.py` (`make monitoring-drift`): `Report` con
  `DataDriftPreset` sobre las 11 variables de entrada, `ValueDrift` del objetivo y
  `DataSummaryPreset`. Los identificadores quedan fuera y la demanda se mide aparte, sin contar en
  la proporción de variables con drift. Los métodos son fijos (Wasserstein normalizada para
  numéricas y Jensen-Shannon para categóricas, umbral 0.1) para que pruebas y datos reales usen la
  misma regla. El conteo de la tabla se verifica contra el `DriftedColumnsCount` de Evidently.
- Tres comparaciones: la oficial contra octubre, el contexto contra abr–sep, y octubre solo con las
  estaciones que el champion vio, para aislar el efecto de las 100 estaciones nuevas.
- Exporta la tabla por variable (CSV) y un resumen (JSON) sin fechas de ejecución, de modo que la
  corrida es reproducible byte a byte. Incluye además calidad por ventana, estaciones nuevas, la
  comparación contra el mismo mes de años previos y tres gráficas.
- `docs/evidencias/drift/`: HTML de Evidently, capturas para el PDF y análisis en `README.md`,
  pensado como insumo de la sección "Monitoreo y drift" del reporte.

## Resultados (datos reales)

| Comparación con la referencia | Variables con drift | Objetivo `demanda` |
|---|---|---|
| Octubre 2026 (oficial) | **7 de 11 (64 %) → drift del conjunto** | 0.298 (+31 %) |
| Abr–sep 2026 | 5 de 11 (45 %) | 0.117 (−11 %) |
| Octubre, solo las 370 estaciones conocidas | 5 de 11; `longitud` 0.141 → 0.012 y `capacidad` 0.108 → 0.012 | 0.501 (+51 %) |

- `mes`/`dia_anio`: cambio esperado con una ventana de un mes, no es un problema de datos.
- `longitud`/`capacidad`: el drift se explica por completo por las estaciones nuevas (21 % de las
  filas de octubre), cuya demanda es baja (17.7 contra 50.6).
- Rezagos y objetivo: en las estaciones conocidas, la demanda de octubre sube **37.9 %** contra
  octubre de 2024 y 2025. Es un cambio de nivel que la estacionalidad no explica y empieza a
  mediados de septiembre.
- Calidad limpia: 0 duplicados, 0 nulos en octubre, mismo esquema y tipos. No apunta a un problema
  de captura.
- El drift no prueba que el modelo empeoró: el MAE relativo de la comprobación de CH1 es parecido
  (18.6 % contra 20.0 %). Lo debe confirmar el A/B de DH4.

## Tipo de cambio

- [x] `feat`: nueva funcionalidad
- [x] `test`: pruebas
- [x] `docs`: documentación

## Cómo se probó

- `make monitoring-data --sin-registry`: huellas idénticas al manifiesto de CH1 (`827ffa8b5727`,
  `4f52f6556fe4`, `b511db2054b2`). El manifiesto versionado no se modifica.
- `make monitoring-drift` sobre datos reales: ver `ejecucion.txt`.
- `python -m pytest tests/test_evidently_report.py -v`: **13 aprobadas** (sin drift, drift
  intencional, objetivo aparte, métodos fijos con < 1,000 filas, esquema, calidad, estaciones
  nuevas, estacionalidad, archivos generados y reproducibilidad). Ver `pytest_evidently.txt`.
- `python -m pytest tests/ -q`: **310 aprobadas**, 1 omitida (E2E del contenedor sin
  `MIBICI_API_URL`).
- `make lint` correcto; `bandit` sin hallazgos en el módulo; `pip-audit -r requirements.txt` sin
  vulnerabilidades conocidas con `evidently==0.7.23`, que no cambia ninguna versión fijada.

## Lista de verificación

- [x] El título del PR sigue Conventional Commits.
- [x] La rama sigue la convención `tipo/descripcion-corta` (`feat/evidently-monitoring`).
- [x] Métodos, umbrales y la carpeta de salida están solo en `mibici/config.py`.
- [x] No se agregan datos, modelos, carpetas de MLflow ni credenciales (los Parquet siguen fuera de Git).
- [x] No se registran datos personales: las tablas solo contienen agregados por variable.
- [x] `evidently==0.7.23` está en `requirements.txt` con versión fija.
- [x] `make lint` y `make test` pasan.
- [x] README actualizado (menú, estructura y sección "Reporte de drift con Evidently").

## Notas para revisión

- Los dos HTML pesan unos 4.4 MB cada uno; se versionan porque la rúbrica pide el reporte de
  Evidently en el repositorio.
- No se toca el registry ni los aliases. El desempeño de A y B queda para el PR de DH4. Este PR
  solo cita la comprobación documentada por CH1.
- El análisis propone, como insumo de Continuous Training, usar el drift del objetivo y del historial
  de demanda como disparador en lugar de la proporción global de columnas, que incluye el
  calendario.

Revisor: HH2 · Horacio.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
