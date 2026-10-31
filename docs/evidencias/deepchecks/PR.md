## Descripción

LH3: los datos de ingesta no tenían un contrato reutilizable que distinguiera
errores de integridad de alertas estadísticas. Se agregan checks DeepChecks para
esquema/tipos, nulos, duplicados, IDs únicos, referencias al catálogo, fechas,
duración y capacidad P2, además de validación temporal de P1/P2.

Una copia con nulos, duplicados, tipos inválidos o estaciones desconocidas produce
condiciones `FAIL`. Una copia desplazada produce `WARN` de drift; el control
idéntico permanece en `PASS`. Los cortes se importan de `config.py`, se detectan
viajes cuyo fin cruza el corte y se prueba que los rezagos de train no usan futuro.
El contrato temporal exige datetime local tanto en inicio como en fin: rechaza
epochs numéricos, texto y fechas con zona horaria para evitar conversiones implícitas.

Base: PR #10 ya integrado en main, commit
`3f1f54d069dd7d2f992987cf59c63d9f35eb92c9`. Se reutilizan sus fixtures,
`compat.py` y `requirements-dev.txt`. Solo se agregan archivos LH3; las
dependencias y los archivos de otros responsables permanecen sin cambios.

## Tipo de cambio

- [x] `feat`: validaciones reutilizables
- [x] `test`: pruebas positivas y anomalías intencionales
- [x] `docs`: evidencia agregada y procedimiento reproducible

## Cómo se probó

Python 3.13.16, DeepChecks 0.19.1, pandas 3.0.6, NumPy 2.5.4 y scikit-learn 1.9.1.

- `python -m pytest tests/test_calidad_datos.py -q`: **52 aprobadas**, sin skips.
- Suite completa: **173 aprobadas**, 109 advertencias, sin fallos ni skips;
  salida en `pytest-suite-final.txt` (609.98 s).
- `python -m ruff check` y `python -m ruff format --check`: correctos.
- Ruff explícito sobre módulo, pruebas y regenerador: correcto.
- `python docs/evidencias/deepchecks/generar.py`: reporte JSON regenerado;
  dos ejecuciones consecutivas producen el mismo SHA-256.
- Una prueba bloquea conexiones de red durante calidad y drift; no se usan
  datos reales, descargas ni MLflow externo.

La ejecución de la suite en el sandbox Windows requiere temporales escribibles:
el comando exacto, los límites de hilos y la advertencia de `pkg_resources`
se documentan en [VALIDACION.md](https://github.com/CristhianHernandez/TC5061-Proyecto-Equipo16/blob/test/deepchecks-datos/docs/evidencias/deepchecks/VALIDACION.md).

## Evidencia e interpretación

[resultados.json](https://github.com/CristhianHernandez/TC5061-Proyecto-Equipo16/blob/test/deepchecks-datos/docs/evidencias/deepchecks/resultados.json) contiene métricas y condiciones de controles y
anomalías, sin muestras ni datasets. Drift KS=0 en control/train-test sintéticos;
KS=1 en el desplazamiento intencional, con umbral configurable de 0.2 y severidad
`WARN`. Es una alerta de distribución, no una afirmación sobre MiBici real ni un
fallo automático del pipeline.

## Revisión

Revisor previsto: **Horacio (HH2)**. Solicitar su revisión al abrir el PR hacia
`main`; aprobación y merge pendientes. No se habilita merge automático.
