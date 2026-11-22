# Resumen FinOps (generado por `scripts/estimate_inference_cost.py`)

Tarifas: AWS us-east-1 (N. Virginia), On-Demand, Linux, facturación por segundo; sin impuestos, planes de ahorro ni capa gratuita, consultadas el 2026-11-21 (USD). Instancia de referencia: `t4g.medium`. Fuente de cada cifra: `references/finops_tarifas.json` y los `manifiesto.json` de la carga.

## Costo de cada ventana de 180 s en la instancia de referencia

| Escenario | Config. API | Pred. exitosas | Pet/s | p95 (ms) | SLO | Costo ventana | Costo / 1,000 | Núcleo-s / inf. | CPU / 1,000 |
|---|---|---|---|---|---|---|---|---|---|
| **u05** | 1 worker, 2 CPU | 869 | 4.8 | 28 | sí | $0.00181 | $0.00208 | 0.0197 | $0.000092 |
| **u20** | 1 worker, 2 CPU | 3,539 | 19.7 | 37 | sí | $0.00185 | $0.00052 | 0.0138 | $0.000064 |
| u60 | 1 worker, 2 CPU | 10,601 | 58.9 | 54 | sí | $0.00247 | $0.00023 | 0.0108 | $0.000051 |
| u120 | 1 worker, 2 CPU | 18,898 | 105.0 | 370 | no | $0.00486 | $0.00026 | 0.0168 | $0.000079 |
| u120_w2 | 2 workers, 2 CPU | 19,805 | 110.0 | 380 | no | $0.00478 | $0.00024 | 0.0156 | $0.000073 |
| u120_w2_cpu4 | 2 workers, 4 CPU | 20,114 | 111.7 | 190 | sí | $0.00915 | $0.00045 | 0.0349 | $0.000163 |
| u120_omp1 | 1 worker, 2 CPU, OMP=1 | 20,678 | 114.9 | 140 | sí | $0.00311 | $0.00015 | 0.0076 | $0.000036 |

## Sensibilidad a la instancia (escenarios oficiales)

| Escenario | Instancia | Arq. | vCPU / GiB | $/h | Cabe con lo medido | Respeta límites del Compose | Costo / 1,000 |
|---|---|---|---|---|---|---|---|
| u05 | t4g.small | arm64 | 2 / 2 | 0.0168 | sí (requiere 0.41 vCPU, 1.68 GiB) | no | $0.00112 |
| u05 | t4g.medium | arm64 | 2 / 4 | 0.0336 | sí (requiere 0.41 vCPU, 1.68 GiB) | sí | $0.00208 |
| u05 | c7g.medium | arm64 | 1 / 2 | 0.0363 | sí (requiere 0.41 vCPU, 1.68 GiB) | no | $0.00224 |
| u05 | c7g.large | arm64 | 2 / 4 | 0.0725 | sí (requiere 0.41 vCPU, 1.68 GiB) | sí | $0.00432 |
| u05 | t3.medium | x86_64 | 2 / 4 | 0.0416 | sí (requiere 0.41 vCPU, 1.68 GiB) | sí | $0.00254 |
| u20 | t4g.small | arm64 | 2 / 2 | 0.0168 | sí (requiere 0.62 vCPU, 1.66 GiB) | no | $0.00029 |
| u20 | t4g.medium | arm64 | 2 / 4 | 0.0336 | sí (requiere 0.62 vCPU, 1.66 GiB) | sí | $0.00052 |
| u20 | c7g.medium | arm64 | 1 / 2 | 0.0363 | sí (requiere 0.62 vCPU, 1.66 GiB) | no | $0.00056 |
| u20 | c7g.large | arm64 | 2 / 4 | 0.0725 | sí (requiere 0.62 vCPU, 1.66 GiB) | sí | $0.00107 |
| u20 | t3.medium | x86_64 | 2 / 4 | 0.0416 | sí (requiere 0.62 vCPU, 1.66 GiB) | sí | $0.00064 |

## Capacidad por instancia de 2 vCPU con SLO p95 ≤ 200 ms

| Config. API | Escenarios | Capacidad (pet/s) | p95 (ms) | Núcleo-s / inf. | RAM máx. API + MLflow | ¿Cota inferior? |
|---|---|---|---|---|---|---|
| 1 worker, 2 CPU | u05 u20 u60 u120 | 58.9 (u60) | 54.0 | 0.0108 | 597 MiB | no |
| 2 workers, 2 CPU | u120_w2 | ninguno cumple | - | - | - | no |
| 2 workers, 4 CPU | u120_w2_cpu4 | 111.7 (u120_w2_cpu4) | 190.0 | 0.0349 | 830 MiB | sí, no se midió más carga |
| 1 worker, 2 CPU, OMP=1 | u120_omp1 | 114.9 (u120_omp1) | 140.0 | 0.0076 | 602 MiB | sí, no se midió más carga |

## Proyección mensual (730 h; utilización objetivo 70%)

| Demanda | Predicciones / mes | Opción | Instancias | Costo / mes | Costo / 1,000 | Utilización |
|---|---|---|---|---|---|---|
| P1 real: lote diario | 14,288 | Actual: VM 24/7 t4g.medium, configuración base | 1 | $26.26 | $1.84 | 0.00923 % |
| P1 real: lote diario | 14,288 | Optimizada: VM 24/7 t4g.small, OMP_NUM_THREADS=1 | 1 | $13.99 | $0.979 | 0.00473 % |
| P1 real: lote diario | 14,288 | Lote: tarea Fargate ARM por ejecución | 1 | $0.23 | $0.0161 | - |
| App pública: 1 M consultas/mes (hipótesis) | 1,000,000 | Actual: VM 24/7 t4g.medium, configuración base | 1 | $26.27 | $0.0263 | 0.646 % |
| App pública: 1 M consultas/mes (hipótesis) | 1,000,000 | Optimizada: VM 24/7 t4g.small, OMP_NUM_THREADS=1 | 1 | $14.01 | $0.014 | 0.331 % |
| Carga sostenida: 60 pet/s (hipótesis) | 157,680,000 | Actual: VM 24/7 t4g.medium, configuración base | 2 | $54.81 | $0.000348 | 50.9 % |
| Carga sostenida: 60 pet/s (hipótesis) | 157,680,000 | Optimizada: VM 24/7 t4g.small, OMP_NUM_THREADS=1 | 1 | $18.82 | $0.000119 | 52.2 % |
