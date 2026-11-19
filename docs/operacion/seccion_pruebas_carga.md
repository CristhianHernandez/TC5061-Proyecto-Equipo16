# Pruebas de carga (sección del reporte · LH3)

> Insumo para `TC5061_Operacion_FinOps_Equipo16.pdf`. Figuras en
> `docs/evidencias/operacion/carga/graficas/`; datos y manifiestos por corrida en
> `docs/evidencias/operacion/carga/<escenario>/`.

## Escenarios y ambiente

Se probó el endpoint `POST /predict` de la API servida por el stack de Docker Compose (API +
MLflow) con Locust 2.47.0 en modo headless. El modelo fue el champion de P1
(`mibici-equipo16-p1-demanda`, versión 2 del registry del equipo, registrada como versión 1 en el
MLflow del stack con `origen = bundle`). Cada usuario virtual envía **un registro por petición**,
rotando en orden fijo 24 registros reales de octubre de 2026, y hace una pausa aleatoria de 0.5 a
1.5 s entre peticiones. Así, cada petición exitosa equivale a una predicción. Una respuesta se
considera exitosa solo si es HTTP 200 y contiene una predicción numérica finita mayor que cero del
modelo esperado; un 200 con otra estructura cuenta como fallo.

Cada corrida tuvo 60 s de calentamiento, que se descartan, y una **ventana estable de 180 s**, con
la misma máquina, imagen y versión del modelo. Los escenarios base fueron **5 y 20 usuarios**
(insumo de FinOps). Para encontrar el punto de degradación se agregaron 60 y 120 usuarios, y a 120
usuarios se probaron tres variantes de configuración. Durante cada ventana se midieron CPU y
memoria de los contenedores de la API y de MLflow con `docker stats`. El stack corrió con sus
límites por defecto (API: 1 worker, 2 CPU, 2 GiB) en un Apple M4 Pro (12 núcleos, 24 GiB) con
Docker Desktop. **Locust corrió en el mismo equipo que el stack**, por lo que compite por CPU,
aunque no emitió avisos de saturación del generador.

## Resultados

| Escenario | Peticiones/s | p50 (ms) | p95 (ms) | Error | CPU API prom. | RAM API |
|---|---|---|---|---|---|---|
| 5 usuarios | 4.8 | 19 | 28 | 0 % (0 de 869) | 10 % | 217 MiB |
| 20 usuarios | 19.7 | 14 | 37 | 0 % (0 de 3,539) | 27 % | 211 MiB |
| 60 usuarios | 58.9 | 16 | 54 | 0 % (0 de 10,601) | 64 % | 211 MiB |
| 120 usuarios | 105.0 | 120 | 370 | 0 % (0 de 18,898) | 177 % | 226 MiB |
| 120 · 2 workers | 110.0 | 30 | 380 | 0 % | 172 % | 440 MiB |
| 120 · 2 workers · 4 CPU | 111.7 | 51 | 190 | 0 % | 390 % | 443 MiB |
| 120 · `OMP_NUM_THREADS=1` | 114.9 | 27 | 140 | 0 % | 88 % | 221 MiB |

*CPU: 100 % = 1 núcleo. MLflow usó 2-3 % de CPU y ~380 MiB en todos los escenarios.*

Figuras sugeridas: `latencia_escalamiento.png` (p50/p95), `rps_escalamiento.png` (peticiones por
segundo), `errores.png` (tasa de error), `recursos_api_escalamiento.png` (CPU/RAM) y
`resumen.png` (p95 y CPU de los siete escenarios).

## Interpretación

**Sin degradación hasta 60 usuarios.** Las peticiones atendidas crecen en proporción a los
usuarios. La mediana se mantiene entre 14 y 19 ms y la p95 sube de forma moderada, de 28 a 54 ms.
La CPU de la API crece casi linealmente y la memoria permanece constante en ~210 MiB, porque el
modelo se carga una sola vez. MLflow no participa en la ruta de inferencia.

**Degradación a 120 usuarios por CPU.** La p50 y la p95 se multiplican por siete (120 y 370 ms).
La CPU de la API promedia 177 %, con percentil 95 de 201 %, es decir, en el límite de 2 CPU del
contenedor. Como Locust es de lazo cerrado, la latencia reduce la tasa atendida a 105 peticiones/s,
frente a ~118 posibles. No hubo errores en ningún escenario: el sistema encola en lugar de
rechazar.

**Causa raíz medida: sobresuscripción de hilos.** El champion es un HistGradientBoosting, cuyo
`predict` paraleliza con OpenMP usando un hilo por núcleo visible: dentro del contenedor hay 12
hilos de OpenMP aunque la cuota sea de 2 CPU. Con un registro por petición, esos hilos consumen
CPU sin acelerar la predicción. Lo confirman tres variantes medidas a 120 usuarios:

- Con 2 workers y la misma cuota baja la p50, pero no la p95, y la memoria se duplica.
- Con 4 CPU, la API consume casi las 4 CPU para la misma carga.
- Con `OMP_NUM_THREADS=1` la CPU baja a la mitad (177 % → 88 %), la p95 a 140 ms y se atiende
  toda la carga ofrecida, con la misma memoria y las mismas predicciones (verificadas como
  idénticas).

**Implicación para operación y costo.** Fijar `OMP_NUM_THREADS=1` en el servicio de la API es la
optimización prioritaria: no tiene costo y reduce a la mitad la CPU por inferencia. Agregar workers
o CPU sin ese ajuste aumenta memoria o consumo sin mejorar la cola de latencia. Falta medir esa
configuración con 1 y 2 workers por encima de 120 usuarios y con 1 CPU, para fijar la capacidad
máxima por instancia antes de recomendar escalado horizontal.
