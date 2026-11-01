# Informe de auditoría de seguridad — MiBici Equipo 16

**Fecha:** 1 de noviembre de 2026  
**Responsable:** HH2  
**Rama:** `security/auditoria-repo`

## 1. Objetivo

Evaluar la seguridad del repositorio mediante una revisión preliminar de archivos sensibles, análisis estático del código Python y auditoría de dependencias, identificando hallazgos y documentando las acciones correctivas.

## 2. Revisión de archivos sensibles

Se revisaron los archivos versionados en busca de archivos `.env`, bases de datos locales de MLflow, artefactos y claves privadas mediante patrones de nombres y extensiones.

La búsqueda identificó únicamente `.env.example`, cuyo contenido no presenta credenciales reales.

Adicionalmente, se buscaron patrones relacionados con contraseñas, claves de API y tokens en archivos Python y de configuración, sin encontrar coincidencias.

**Limitación:** Estas comprobaciones no constituyen una auditoría exhaustiva de secretos ni incluyen una revisión completa del historial de Git.

## 3. Análisis estático con Bandit

**Herramienta:** Bandit 1.9.4  
**Comando:** `python -m bandit -r mibici`

| Indicador | Resultado |
|---|---:|
| Líneas analizadas | 3,013 |
| Hallazgos bajos | 5 |
| Hallazgos medios | 0 |
| Hallazgos altos | 0 |
| Archivos omitidos | 0 |

Los cinco hallazgos se localizaron en `mibici/tracking.py`:

- **B404:** Importación del módulo `subprocess` (1 hallazgo).
- **B603:** Ejecución de procesos externos (2 hallazgos).
- **B607:** Ejecución mediante una ruta parcial al ejecutable Git (2 hallazgos).

Los comandos analizados utilizan argumentos constantes y no emplean `shell=True`, lo que reduce el riesgo de inyección de comandos.

Sin embargo, la resolución de `git` mediante `PATH` conserva un riesgo residual si el entorno de ejecución está comprometido.

**Tratamiento:** Los hallazgos se mantienen documentados para evaluación posterior. No se introdujeron exclusiones `#nosec`.

## 4. Auditoría de dependencias con pip-audit

**Herramienta:** pip-audit 2.10.1  
**Comando:** `python -m pip_audit -r requirements.txt`

### Hallazgo inicial

| Elemento | Resultado |
|---|---|
| Paquete | `mlflow` |
| Versión afectada | 3.17.1 |
| Identificador | CVE-2026-53995 |
| Versión corregida reportada | 3.17.2 |

### Acción correctiva

Se modificó `requirements.txt` para actualizar la dependencia:

`mlflow==3.17.1` → `mlflow==3.17.2`

Se instaló la nueva versión en el entorno virtual y se repitió la auditoría.

**Resultado posterior:** `No known vulnerabilities found`.

El paquete local `mibici (0.0.1)` no pudo auditarse contra PyPI. Esta limitación quedó registrada en la evidencia.

## 5. Validación funcional

Se ejecutó:

`python -m pytest tests/test_tracking.py -v --disable-warnings`

**Resultados:**

- 9 pruebas aprobadas.
- 0 pruebas fallidas.
- 1 advertencia.
- Tiempo de ejecución: 11.89 segundos.

Las pruebas cubren funcionalidades del seguimiento de experimentos con MLflow, incluyendo registro de métricas, parámetros, etiquetas y ejecuciones anidadas.

La validación no sustituye una prueba de regresión completa del repositorio.

## 6. Riesgos residuales y pendientes

- Evaluar el riesgo asociado con la resolución del ejecutable Git mediante `PATH`.
- Completar la revisión de posibles secretos en otros formatos y en el historial del repositorio.
- Coordinar la actualización de dependencias con CH1.
- Ejecutar pruebas adicionales de integración o regresión si el equipo lo considera necesario.

## 7. Conclusión

La auditoría identificó cinco hallazgos de severidad baja mediante Bandit y una vulnerabilidad conocida en MLflow mediante pip-audit.

Se actualizó MLflow a la versión 3.17.2. La auditoría posterior no identificó vulnerabilidades conocidas entre las dependencias auditables y las nueve pruebas del módulo de seguimiento de experimentos finalizaron correctamente.

Los hallazgos residuales y las limitaciones de cobertura permanecen documentados para seguimiento.

## 8. Evidencias

- `docs/evidencias/seguridad/bandit_reporte.txt`
- `docs/evidencias/seguridad/pip_audit_reporte.txt`
- `requirements.txt` — actualización de MLflow
