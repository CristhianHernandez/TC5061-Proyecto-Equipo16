# LH3 · Calidad e integridad con DeepChecks

Responsable: Luis (LH3). Revisor previsto: Horacio (HH2).
Rama: `test/deepchecks-datos`. Base: `main` en
`3f1f54d069dd7d2f992987cf59c63d9f35eb92c9`, que integra el
[PR #10 de CH1](https://github.com/CristhianHernandez/TC5061-Proyecto-Equipo16/pull/10).
La integración se confirmó en GitHub y se ejecutó `git pull --ff-only origin main`
antes de crear la rama. La revisión de HH2 sigue pendiente.

## Alcance y ejecución

Se reutilizan `mibici/compat.py`, las versiones de `requirements-dev.txt` y los
fixtures de `tests/conftest.py`. No se cambian dependencias, README, seguridad,
ingesta, features ni pruebas de otros integrantes. Los archivos locales previos
`.vscode/` y el PDF del plan quedan fuera de esta contribución.

Desde la raíz, con Python 3.13 y las dependencias de desarrollo instaladas:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest tests/test_calidad_datos.py -q
python -m pytest tests/ -q
python -m ruff check
python -m ruff format --check
# El include de pyproject.toml no abarca tests/ ni docs/: comprobarlos explícitamente.
python -m ruff check mibici/validation tests/test_calidad_datos.py docs/evidencias/deepchecks/generar.py
python -m ruff format --check mibici/validation tests/test_calidad_datos.py docs/evidencias/deepchecks/generar.py
python docs/evidencias/deepchecks/generar.py
```

La ejecución local usa `.venv/Scripts/python.exe` (Python 3.13.16) y las versiones
registradas en `resultados.json`: DeepChecks 0.19.1, NumPy 2.5.4, pandas 3.0.6,
scikit-learn 1.9.1, PyTest 9.1.1 y Ruff 0.17.0.

Resultados finales ejecutados el 30 de octubre de 2026:

| Verificación | Resultado |
| --- | --- |
| Pruebas LH3 | 52 aprobadas, 1 advertencia, 4.05 s |
| Suite completa | 173 aprobadas, 109 advertencias, 609.98 s |
| Ruff del repositorio | Check correcto; 18 archivos con formato correcto |
| Ruff explícito de archivos LH3 | Check correcto; 4 archivos con formato correcto |
| Regeneración JSON | Dos ejecuciones consecutivas con el mismo SHA-256 |

No hubo fallos, errores ni omisiones en la ejecución final. Las advertencias se
contabilizan en las salidas; no se convierten en fallos ni se deshabilitan reglas
de validación. El único aviso de las pruebas LH3 es la deprecación de
`pkg_resources` explicada más adelante. El SHA-256 de `resultados.json` es
`5a7498bd1ea261c66d5e994f31e0fbdc78b4886eb8d36cb0024d1cc33c66efd1`.

En el sandbox Windows, el directorio temporal predeterminado denegó la creación
de carpetas PyTest (primera suite: 124 aprobadas, 40 errores de setup). Para
reproducir la suite dentro de un directorio escribible se usó:

```powershell
$lh3Temp = Join-Path (Get-Location).Path '.venv\tmp'
New-Item -ItemType Directory -Force -Path $lh3Temp | Out-Null
$env:TEMP = $lh3Temp
$env:TMP = $lh3Temp
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
python -m pytest tests/ -q --disable-warnings --tb=short --basetemp=.venv/pytest-lh3-revision-final
```

`--basetemp` es exclusivo para pruebas y PyTest puede recrear esa carpeta. Los
límites de hilos evitan sobrecarga en los modelos pequeños de la suite existente.
Los fixtures de MLflow usan SQLite y artefactos temporales; LH3 no inicia corridas.
No se usan MiBici remoto, datos personales ni el servidor MLflow del equipo.

## Dimensiones y hallazgos

Todos los contratos devuelven `CheckResult` con condiciones evaluadas por
DeepChecks. `EsquemaColumnas`, `IntegridadViajes`, `IntegridadEstaciones` e
`IntegridadTemporal` extienden su API `SingleDatasetCheck`; `PercentOfNulls`,
`DataDuplicates`, `MixedDataTypes` y `FeatureDrift` son checks nativos.

| Dimensión | Regla / check | Evidencia negativa |
| --- | --- | --- |
| Esquema y tipos | Columnas reales de `dataset.py`; enteros, fechas locales, reales, texto y booleanos | IDs convertidos a texto; fecha con texto inválido; mezcla de texto y números |
| Completitud | `PercentOfNulls`, tolerancia cero en columnas de ingesta | Un `pd.NA` en cada columna de viajes, probado por separado |
| Duplicados | `DataDuplicates`, proporción cero | Una fila añadida: `1/201` duplicados |
| Unicidad de IDs | `viaje_id` y `estacion_id` únicos | Mismo ID con contenido diferente: el check de duplicados exactos da cero, integridad falla |
| Referencias | Ambos extremos pertenecen al catálogo | Una estación inexistente en origen y en destino, por separado |
| Fechas y duración | Inicio dentro del periodo; fechas interpretables; duración `(0, 1440]` minutos, como `limpiar_viajes` | Inicio fuera del periodo, `NaT`, fecha inválida, duración negativa, cero o superior a 24 h |
| Catálogo y P2 | Coordenadas válidas; capacidad entera no negativa; capacidad estrictamente positiva para P2 | Capacidad negativa, faltante, infinita, fraccionaria o cero en catálogo P2 |
| Partición temporal | Intervalos de `config.py`, claves no repetidas y fechas no nulas | Fila test en train, fecha fuera de periodo, partición vacía, clave repetida y viaje cuyo fin cruza el corte |
| Drift | `FeatureDrift` sobre predictores numéricos explícitos | Copia con desplazamiento de +10000: KS = 1, condición `WARN` |

El catálogo compartido tiene seis estaciones, una inactiva y otra con capacidad
cero. Ambas pueden existir en la ingesta; antes de P2 se excluye capacidad cero,
como hace `p2_riesgo.preparar_conjuntos`. No se inventa inventario observado.

Los nulos de rezagos iniciales de P1/P2 son esperables y se imputan en sus modelos;
el contrato estricto de nulos aquí corresponde a **ingesta**, no a todas las
features. Para drift se eligen dos predictores sin nulos. La API rechaza nulos e
infinitos antes del cálculo, sin decidir una imputación usando test.

## Cortes y ausencia de fuga

Las funciones y pruebas importan los cortes de `config.py`; no definen otros:

| Conjunto | Intervalo local |
| --- | --- |
| train | `[FECHA_INICIO, INICIO_VALIDACION)` |
| val | `[INICIO_VALIDACION, INICIO_PRUEBA)` |
| test | `[INICIO_PRUEBA, FECHA_FIN + 1 día)` |

Se verifican los conjuntos reales construidos por los fixtures P1 y P2, y los
instantes exactos de corte, incluyendo la última hora del último día. Las claves
por defecto son `(estacion_id, fecha)` o `(estacion_id, fecha_franja)`, permitiendo
que una estación aparezca en distintos periodos. Para viajes se usa `viaje_id` y,
opcionalmente, `columna_fin='fin'` para detectar información conocida después
del corte de su partición. Esta validación informa el cruce; no mueve ni elimina filas.
Tanto la fecha de inicio como la de fin deben tener tipo datetime sin zona,
conforme a la ingesta. La revisión adicional detectó y corrigió que `fin` numérico
podía interpretarse como un epoch válido. Se agregaron seis regresiones para
inicio/fin con enteros, texto o zona horaria; estas entradas ahora se rechazan.

La prueba de causalidad altera únicamente demanda futura y comprueba que los
rezagos semanales y las medias móviles de train permanecen idénticos y contienen
valores calculados. Esto complementa el contrato de asignación temporal; no es
una certificación de todas las posibles features de un modelo futuro.

## Interpretación y compatibilidad

En los fixtures, train frente a test y el control idéntico dan KS = 0 para ambos
predictores. El desplazamiento intencional da KS = 1 y `WARN`, usando umbral
operativo configurable de 0.2 sobre la **magnitud** KS; no es un p-valor.
Los datos sintéticos no permiten concluir que MiBici real carezca de drift.
Una alerta exige revisar estacionalidad, composición y métricas del modelo;
no demuestra por sí sola un error de código ni obliga a reentrenar.

Para evaluar condiciones sin convertir advertencias en fallos:

```python
resultado = evaluar_drift(train, test, ["demanda_lag_1", "demanda_media_7"])
assert resultado.passed_conditions(fail_if_warning=False)
```

Las violaciones de integridad usan `FAIL`; los controles válidos usan `PASS`.
El check nativo de duplicados trae una condición `WARN` por defecto: se sustituye
por una condición explícita de cero duplicados que devuelve `FAIL`.

Se aplican los alias y el scorer de `mibici/compat.py`; las copias nullable para
checks nativos pasan por `a_tipos_numpy`. El esquema se evalúa **antes** de esa
conversión, para no ocultar tipos incorrectos. No se ejecutan las suites generales
con `OutlierSampleDetection`, `TrainTestSamplesMix` o `MultivariateDrift`, cuyas
incompatibilidades ya documenta `compat.py`. Los checks seleccionados y el drift
sí se ejecutan, sin skips por dependencia. Sigue vigente el aviso de deprecación
de `pkg_resources`; `setuptools==80.10.2` ya estaba fijado por CH1.

La importación desactiva la telemetría y la consulta de versión de DeepChecks
mediante sus variables de entorno oficiales. Una prueba bloquea conexiones de
socket y comprueba que las validaciones y el drift funcionan sin red.

## Evidencia versionable

- `resultados.json`: métricas y condiciones agregadas de casos sanos y alterados,
  versiones, tamaños y cortes. No incluye filas, ejemplos, HTML ni objetos de modelo.
- `generar.py`: regenerador que invoca los proveedores de los fixtures compartidos
  mediante `__wrapped__`, sin iniciar los fixtures MLflow. Se ejecuta desde la raíz.
- `pytest-lh3.txt`, `pytest-suite-final.txt` y `ruff.txt`: salidas ligeras de verificación;
  `--disable-warnings` reduce el ruido de los logs, sin cambiar los resultados.
- `PR.md`: descripción preparada para revisión de HH2.

Los archivos de entorno, datasets y binarios de las pruebas permanecen en `.venv`
y no forman parte del PR. La instalación utilizó exactamente `requirements-dev.txt`;
no fue necesario cambiar versiones ni coordinar una nueva dependencia con CH1.
