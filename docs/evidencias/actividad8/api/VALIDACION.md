# Actividad 8 · LH3 · API FastAPI/Pydantic

Responsable: Luis (LH3). Revisor: Horacio (`HHSavenger`). Verificación local del
5 de noviembre de 2026, con Python 3.13.16.

## Base y dependencia de CH1

Base actual: `main` en `f4eebfc6c8c56606195aff1f81abf96982796158`, que ya
incluye el PR #15 de Cristhian por squash y las contribuciones de la Actividad 7.
El loader y su contrato pertenecen a CH1 y se reutilizan desde main sin modificarlos.

La rama local `feat/fastapi-predict` está alineada con ese main; su PR puede
dirigirse directamente a **main**, con Horacio (`HHSavenger`) como revisor.
Los archivos locales previos `.vscode/` y el PDF del plan se conservaron.

Para revisar el alcance de Luis:

```powershell
git diff main
git status --short
```

Los archivos nuevos sin seguimiento también se muestran con `git status`;
`git diff` por sí solo no los incluye. No usar `git add .`: añadir solo los
archivos LH3 indicados al final de este documento.

## Contrato y comportamiento

La entrada es `{"registros": [{...}, {...}]}`, con uno o más registros. Los
nombres, orden, tipos MLflow y obligatoriedad provienen del contrato de CH1:
[`docs/serving/contrato_p1.md`](../../../serving/contrato_p1.md).

| Variables | Validación HTTP |
| --- | --- |
| `dia_semana`, `mes`, `dia_anio` | Enteros estrictos, rangos 0–6, 1–12 y 1–366 |
| `es_fin_semana`, `es_festivo` | Enteros 0/1; fin de semana coherente con `dia_semana` |
| `demanda_lag_1`, `demanda_lag_7`, `demanda_media_7` | Números finitos no negativos; opcionales o `null` |
| `capacidad` | Entero positivo representable por la firma int32 |
| `latitud`, `longitud` | Números finitos en [-90, 90] y [-180, 180] |

Se rechazan campos adicionales, listas vacías, campos obligatorios ausentes,
booleanos usados como números, texto numérico e infinitos/NaN. No se convierten
silenciosamente estos valores a números. Los rangos observados en train no se
usan como límites duros: una capacidad 50 o demanda pasada 1000 no es inválida
por superar el máximo histórico. No se inventa un polígono metropolitano sin
un catálogo geográfico que lo defina.

El loader conserva el preprocesamiento registrado. La API recibe las features
ya calculadas: no transforma viajes ni calcula rezagos con el conjunto de prueba.
Al iniciar se contrasta la firma cargada con las 11 variables de P1 features-v1;
un modelo con otro contrato, por ejemplo features-v2, no se declara listo.

| Ruta / situación | Resultado |
| --- | --- |
| `GET /health`, modelo y firma cargados | 200, `estado=listo`, metadatos públicos |
| `GET /health`, error de carga o contrato incompatible | 503, `estado=no_disponible` |
| `POST /predict`, entrada y predicción válidas | 200, un float finito positivo por registro, en el mismo orden |
| Entrada inválida / `ErrorEntrada` del loader | 422 |
| Modelo ausente / `ModeloNoDisponible` | 503; la pérdida del modelo también invalida readiness |
| Excepción de inferencia / salida incorrecta | 500, mensaje fijo sin detalles internos |
| `GET /docs` y `/openapi.json` | Documentación y esquemas automáticos, incluso si falla la carga |

La salida se valida con `response_model`; se convierten escalares NumPy a
floats JSON y se rechazan resultados multidimensionales, longitudes incorrectas,
texto, booleanos, cero, negativos, NaN e infinitos. No se corrige ni se recorta
la predicción para ocultar errores del modelo.
Antes de convertir a float se conservan los tipos originales: una lista mixta
como `[True, 2.5]` se rechaza y no se acepta como `[1.0, 2.5]` por promoción de NumPy.

Solo se publican `nombre`, `version`, `alias` y `run_id`. No se devuelve el
diccionario completo de metadatos: puede contener URIs, rutas y tags internos.
Las respuestas 422 tampoco reflejan el payload ni `ctx` de Pydantic; esto permite
devolver JSON válido incluso ante NaN/Infinity escritos en el cuerpo recibido.

Se usa lifespan para cargar una vez por proceso, y las operaciones síncronas del
modelo se ejecutan fuera del event loop. Si falla el arranque del modelo, la app
permanece disponible para diagnóstico con `/health=503`; corregir la configuración
y reiniciar. No hay recarga del alias en cada petición.
Cada petición conserva el modelo y sus metadatos al comenzar la inferencia; si
otra petición declara el modelo no disponible, la que ya estaba en curso puede
terminar con sus propios metadatos y las siguientes reciben 503.

Referencias de implementación: [lifespan de FastAPI](https://fastapi.tiangolo.com/advanced/events/),
[manejo de errores](https://fastapi.tiangolo.com/tutorial/handling-errors/) y
[modo estricto de Pydantic](https://docs.pydantic.dev/latest/concepts/strict_mode/).

## Instalar y ejecutar

Con un entorno Python 3.13 activo, desde la raíz del repositorio:

En esta carpeta se utilizó `.venv/Scripts/python.exe`; puede usarse esa ruta en
lugar de `python` sin activar el entorno ni cambiar el Python global de Anaconda.

```powershell
python -m pip install -r requirements-dev.txt -r requirements-api.txt
python -m pytest tests/test_api.py -q
python -m uvicorn mibici.serving.api:app --host 127.0.0.1 --port 8000
```

`requirements-api.txt` fija únicamente las dependencias de esta capa HTTP:
FastAPI 0.162.0, Starlette 1.10.0, Pydantic 2.14.2, Uvicorn 0.54.0 y HTTPX 0.28.1
para las pruebas. Se fija Starlette para conservar la compatibilidad de TestClient
con HTTPX que se verificó, hasta que el equipo acuerde migrar a HTTPX2.
No se editan `requirements.txt` ni `requirements-dev.txt`; HH2 podrá reutilizar
estas versiones al definir el runtime de la imagen. Se actualizó únicamente el
entorno `.venv` a MLflow 3.17.2, versión ya fijada en main por seguridad.

La estrategia de producción sigue siendo la de CH1: `MODEL_PATH` con un bundle
exportado del Registry. Si falta, el loader usa `MODEL_URI` y `MLFLOW_TRACKING_URI`.
Para el Registry SQLite local verificado en esta carpeta:

```powershell
$env:MLFLOW_TRACKING_URI = 'sqlite:///E:/TEC/MLOps/TC5061-Proyecto-Equipo16/mlflow.db'
python -m uvicorn mibici.serving.api:app --host 127.0.0.1 --port 8000
```

En otra terminal, desde la raíz:

```powershell
curl.exe -i http://127.0.0.1:8000/health
curl.exe -i -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data-binary "@docs/serving/ejemplo_entrada_p1.json"
```

La ruta SQLite anterior es un comando local de preparación; no se incorpora al
código de la API ni se propone como ruta para el contenedor. Cambiarla según el
equipo. Para la imagen usar el bundle exportado por CH1; su build y prueba real
corresponden a HH2.

## Pruebas y evidencia reproducible

Las pruebas API usan `TestClient` con `ModeloServido` y `preparar_entrada` reales
de CH1; solo se sustituye el predictor por una regla determinista. No entrenan,
no descargan datos y bloquean los transportes HTTP externos. Incluyen:

- 200, múltiples registros, orden de columnas/registros y dtypes de la firma.
- Rezagos omitidos o nulos equivalentes y ausencia de mutación de entradas.
- 422 por rangos, campos faltantes/adicionales, tipos, cuerpo o JSON inválidos.
- Salud, Swagger, OpenAPI, carga única y limpieza de estado al cerrar lifespan.
- 503 ante carga fallida, firma incompatible o pérdida del modelo.
- 500 ante fallos de inferencia y salidas inválidas; mensajes privados no filtrados.
- Conversión NumPy y serialización finita del resultado.
- Rechazo de booleanos en salidas mixtas y metadatos conservados en peticiones en curso.

En el sandbox de Windows, `asyncio` se bloqueó en `socketpair` al iniciar
`TestClient`: su implementación usa loopback incluso para la comunicación
interna del event loop. La ejecución fuera del sandbox pasó; no se cambió el
backend ni se omitieron tests. El guard de red de los tests bloquea los
transportes HTTP, sin bloquear esos sockets internos de Windows.

Pruebas específicas ejecutadas: **66 aprobadas**, 1 advertencia, 0.78 s.
Suite completa con la base actualizada, el loader de CH1 y la API de LH3:
**279 aprobadas**, 113 advertencias, 450.31 s. Sin fallos, errores ni omisiones.
Esa suite se ejecutó en la revisión de código anterior. El árbol de archivos de
CH1 probado entonces es idéntico al del squash #15 ahora integrado en main.
Ambos árboles tienen SHA `9f53899a2726a95e1e398065c9bf683018e063b6`.
Después de alinear la rama se volvieron a ejecutar las pruebas API, Ruff y la
petición real por Uvicorn; no se alteró la implementación para cambiar de base.
Ruff del repositorio y de los cuatro archivos Python nuevos: correcto; formato
verificado para 23 archivos del paquete y los cuatro archivos LH3 explícitos.
Resultados y logs: `pytest-api.txt`, `pytest-suite.txt`, `ruff.txt`.
Existe una advertencia de Starlette sobre la futura sustitución de HTTPX por
HTTPX2 en TestClient. Las versiones fijadas se ejecutan correctamente actualmente;
no se oculta ese aviso ni se agrega una migración de dependencias fuera del alcance.

La suite completa se ejecutó fuera del sandbox con temporales dentro de `.venv`:

```powershell
$lh3Temp = Join-Path (Get-Location).Path '.venv\tmp'
New-Item -ItemType Directory -Force -Path $lh3Temp | Out-Null
$env:TEMP = $lh3Temp
$env:TMP = $lh3Temp
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
python -m pytest tests/ -q --disable-warnings --tb=short --basetemp=.venv/a8-review
```

PyTest puede recrear la carpeta indicada por `--basetemp`; reservarla solo para
estas pruebas. `--disable-warnings` abrevia el log, pero conserva el conteo de
advertencias y no modifica el resultado de los tests.

Para repetir la petición **HTTP real a Uvicorn** y regenerar su JSON:

```powershell
$env:MLFLOW_TRACKING_URI = 'sqlite:///E:/TEC/MLOps/TC5061-Proyecto-Equipo16/mlflow.db'
python docs/evidencias/actividad8/api/comprobar_local.py
```

El script usa un puerto efímero de loopback y detiene el servidor al finalizar.
Lee el payload de CH1 y guarda `respuesta_local.json`, sin modelos ni datasets.
Verifica 200 en health/predict/docs, 422 ante capacidad ausente y la igualdad
entre las predicciones HTTP y las del loader.

**Modelo realmente verificado:** `mibici-equipo16-p1-demanda@champion`, versión
**1 del Registry local**, con la firma esperada de 11 variables. El documento de
CH1 informa versión 2 en su Registry: los números de versión son propios de cada
Registry y no se asume que coincidan. No se reentrenó, promovió ni modificó un
alias. El modelo avisa que fue guardado con MLflow 3.17.1; cargó y predijo con
3.17.2. No se revierte la corrección de seguridad para eliminar ese aviso.

Predicciones reales obtenidas para los dos registros del ejemplo:
`[40.697957479000046, 276.77983750628846]`.
Esta evidencia valida la API local; la prueba en contenedor y su captura final
siguen siendo responsabilidades de HH2/DH4.

## Preparación del PR

Añadir únicamente los archivos propios de LH3:

```powershell
git add mibici/serving/api.py mibici/serving/schemas.py tests/test_api.py requirements-api.txt docs/evidencias/actividad8/api
git commit -m "feat: expone predicciones P1 con FastAPI"
git push -u origin feat/fastapi-predict
```

CH1 ya está integrado por el PR #15. Abrir el PR de LH3 directamente hacia main:

```powershell
gh pr create --base main --head feat/fastapi-predict --title "feat: API FastAPI para predicciones P1" --body-file docs/evidencias/actividad8/api/PR.md --reviewer HHSavenger
```

Esperar la revisión y aprobación antes de hacer merge.

La descripción preparada está en `PR.md`; el revisor de LH3 es `HHSavenger`.
