# Análisis FinOps (sección del reporte · DH4)

> Insumo para `TC5061_Operacion_FinOps_Equipo16.pdf`. Todas las cifras salen de
> `make finops` ([`scripts/estimate_inference_cost.py`](../../scripts/estimate_inference_cost.py)),
> que lee los manifiestos de la carga (PR #2) y las tarifas de
> [`references/finops_tarifas.json`](../../references/finops_tarifas.json). Tablas completas en
> [`docs/evidencias/operacion/finops/resumen.md`](../evidencias/operacion/finops/resumen.md).

## Método

**Fórmula acordada.** Costo de la ventana = horas facturables × tarifa por hora + otros costos
atribuibles de la ventana. Costo por inferencia = costo de la ventana ÷ predicciones exitosas de
esa misma ventana. Cada petición de Locust lleva un registro, así que predicciones exitosas =
peticiones exitosas. El script verifica que peticiones = fallidas + exitosas en cada manifiesto y
que coinciden con `ventana.json`. Si una ventana tuviera cero predicciones exitosas, el costo
unitario queda como *no definido* en lugar de dividir entre cero.

**Factores incluidos.**

| Factor | Valor | Fuente |
|---|---|---|
| Tiempo | Ventana estable de 180 s = 0.05 h facturables (Linux se factura por segundo) | `manifiesto.json` |
| Cómputo | `t4g.medium` (Graviton, arm64, 2 vCPU, 4 GiB): **$0.0336/h** | AWS EC2 On-Demand, us-east-1 |
| Créditos de CPU | $0.04 por vCPU-h sobre la línea base de 20 % por vCPU (modo Unlimited) | AWS EC2 |
| Disco | 20 GB gp3 a $0.08/GB-mes | AWS EBS |
| Registro de imagen | 1.3 GB a $0.10/GB-mes | AWS ECR |
| Transferencia | 162 bytes por respuesta a $0.09/GB | AWS EC2 |
| Escalabilidad | Capacidad con SLO p95 ≤ 200 ms y error ≤ 1 %, utilización objetivo 70 % | Escenarios de carga |

Tarifas On-Demand en USD, consultadas el 2026-11-21, sin impuestos, planes de ahorro ni capa
gratuita.

**Instancia de referencia.** Se midió en arm64 (Apple M4 Pro), y Graviton es la misma
arquitectura, así que la imagen corre sin cambios. La `t4g.medium` respeta los límites que hoy
declara el Compose (API 2 CPU/2 GiB y MLflow 1 CPU/1 GiB). En u05 y u20 la CPU promedio
(0.13 y 0.30 núcleos) queda bajo su línea base de 0.4 núcleos, así que no se cobran créditos.

## Costo por inferencia medido

| Escenario | Predicciones exitosas | Pet/s | Costo de la ventana | Costo por inferencia | Costo por 1,000 |
|---|---|---|---|---|---|
| **u05** (5 usuarios) | 869 | 4.8 | $0.00181 | $0.0000021 | **$0.00208** |
| **u20** (20 usuarios) | 3,539 | 19.7 | $0.00185 | $0.00000052 | **$0.00052** |
| u60 | 10,601 | 58.9 | $0.00247 | $0.00000023 | $0.00023 |
| u120_omp1 | 20,678 | 114.9 | $0.00311 | $0.00000015 | $0.00015 |

Más del 90 % del costo de la ventana es cómputo ($0.00168). Disco y registro suman $0.00012 y la
transferencia menos de $0.00006. Entre u05 y u20 el costo de la ventana casi no cambia, pero se
atienden 4 veces más predicciones, así que el costo unitario baja 4 veces. **Con una VM de precio
fijo, el costo por inferencia depende de la utilización, no del tráfico.**

**Costo por CPU consumida.** Si se cobrara solo la CPU que usa la API (núcleo-segundos por
inferencia × $0.0168 por vCPU-h), u20 costaría $0.000064 por 1,000: 8 veces menos que la ventana.
Esa diferencia es capacidad ociosa pagada. La CPU por inferencia baja con la carga (0.0197 →
0.0138 → 0.0108 núcleo-s de u05 a u60) porque el costo fijo del proceso se reparte, y sube al
saturar (0.0168 en u120).

**Sensibilidad a la instancia (u20).** `t4g.small` $0.00029 · `t4g.medium` $0.00052 ·
`c7g.medium` $0.00056 · `t3.medium` (x86) $0.00064 · `c7g.large` $0.00107 por 1,000. Con lo
medido (RAM máxima de API + MLflow ≈ 0.6 GiB, más 768 MiB de sistema y Docker con 25 % de holgura
= 1.7 GiB; CPU p95 de 0.5 núcleos al 80 % de utilización = 0.62 vCPU) todas caben, incluida la
`t4g.small`. Sin embargo, la `t4g.small` no respeta los límites actuales del Compose.

## Proyección de operación mensual (730 h facturadas)

| Demanda | Pred./mes | Despliegue | Costo/mes | Por 1,000 | Utilización |
|---|---|---|---|---|---|
| P1 real: lote diario (470 estaciones) | 14,288 | VM 24/7 `t4g.medium` actual | $26.26 | $1.84 | 0.009 % |
| | | VM 24/7 `t4g.small` + `OMP_NUM_THREADS=1` | $13.99 | $0.98 | 0.005 % |
| | | Tarea Fargate ARM por ejecución (lote) | **$0.23** | **$0.016** | - |
| App pública (hipótesis) | 1,000,000 | VM 24/7 `t4g.medium` actual | $26.27 | $0.026 | 0.65 % |
| | | VM 24/7 `t4g.small` + OMP=1 | $14.01 | $0.014 | 0.33 % |
| 60 pet/s sostenidas (hipótesis) | 157,680,000 | 2 × `t4g.medium` actual | $54.81 | $0.00035 | 51 % |
| | | 1 × `t4g.small` + OMP=1 | $18.82 | $0.00012 | 52 % |

**Hallazgo principal.** El uso real de P1 es una predicción por estación al día: 14,288 al mes.
Una VM encendida todo el mes las atiende con 0.009 % de utilización. Así, la inferencia cuesta
$1.84 por 1,000, **3,500 veces más que en la prueba de carga** ($0.00052). La prueba mide lo que
cuesta el sistema lleno; la factura depende de cuántas horas está encendido.

**Escalabilidad.** Con la configuración actual, una instancia de 2 vCPU cumple el SLO hasta
58.9 pet/s (u60). A 120 usuarios, la p95 sube a 370 ms. Con `OMP_NUM_THREADS=1` atiende
114.9 pet/s con p95 de 140 ms. Esa cifra es una cota inferior, porque no se midió más carga. Para
60 pet/s sostenidas se pasa de 2 instancias a 1. La `t4g.small` sí paga créditos de CPU en ese
caso ($2.53/mes), porque usa 0.49 núcleos frente a una línea base de 0.4, y aun así es la opción
más barata.

## Optimizaciones priorizadas

| # | Acción | Evidencia | Impacto estimado |
|---|---|---|---|
| 1 | **Servir P1 en lote**: tarea programada que arranca, predice las 470 estaciones y se apaga | Demanda real 14,288 pred/mes; utilización de la VM 0.009 % | $26.26 → $0.23/mes (−99 %) |
| 2 | **`OMP_NUM_THREADS=1` y `OPENBLAS_NUM_THREADS=1`** en el servicio `api` | u120 → u120_omp1: CPU por inferencia 0.0168 → 0.0076 núcleo-s (−55 %), p95 370 → 140 ms, mismas predicciones | Duplica la capacidad por instancia; a 60 pet/s pasa de $54.81 a $18.82/mes (−66 % junto con #3) |
| 3 | **Ajustar el tamaño**: bajar `API_MEMORIA` y `MLFLOW_MEMORIA` (p. ej. 768m y 512m) y usar `t4g.small` | RAM máxima medida: API 226 MiB y MLflow 392 MiB | VM 24/7: $26.26 → $13.99/mes (−47 %) |
| 4 | Ya aplicada: `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false` | MLflow en reposo 1.6 GiB → 360 MiB (PR #1) | Sin este ajuste, el stack no cabría en 2 GiB y #3 no sería posible |
| 5 | Sacar MLflow de la instancia de servicio | La API carga el modelo una vez; MLflow usa 2-3 % de CPU y ~0.38 GiB solo para estar disponible | Libera ~63 % de la RAM del stack. No se cuantifica porque requiere un MLflow compartido |
| 6 | Descartadas por datos: 2 workers o 4 CPU sin fijar hilos | 2 workers: RAM ×2 y p95 igual (380 ms). 4 CPU: 0.0349 núcleo-s por inferencia (×2.1) y costo por 1,000 ×1.8 | Aumentan el costo sin mejorar la latencia de cola |

Para un servicio 24/7 con carga estable, un Savings Plan o una instancia reservada reduciría la
tarifa por hora. No se cuantifica porque depende del plazo de compromiso.

## Límites y qué volver a medir

- Se midió en un M4 Pro local con Locust en el mismo equipo. Un núcleo Graviton2 de `t4g` es más
  lento, así que la capacidad en la nube puede ser menor. Las cifras de nube son una
  **extrapolación**.
- La variante de 4 CPU se costeó en la instancia de referencia de 2 vCPU solo para comparar la CPU
  consumida. En la nube requeriría una instancia de 4 vCPU.
- Antes de recomendar escalado, volver a medir:
  - `OMP_NUM_THREADS=1` con 1 y 2 workers a 120, 200 y 300 usuarios.
  - `API_CPUS=1` con memoria reducida.
  - Una `t4g.small` real, con Locust en otra máquina.
  - Comparar arm64 contra x86 (`t3.medium`).
