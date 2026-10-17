## Descripción

<!-- ¿Qué cambia este PR y por qué? Relaciónalo con la pregunta (P1, P2, P3) o entregable que atiende. -->

## Tipo de cambio

- [ ] `feat`: nueva funcionalidad
- [ ] `fix`: corrección de error
- [ ] `docs`: documentación
- [ ] `refactor`: reestructura sin cambiar comportamiento
- [ ] `test`: pruebas
- [ ] `chore`: mantenimiento (dependencias, configuración)

## Cómo se probó

<!-- Comandos ejecutados y resultado (p. ej. `make lint`, `make test`, `python -m mibici.dataset`). -->

## Lista de verificación

- [ ] El título del PR sigue Conventional Commits (`tipo: descripción`).
- [ ] La rama sigue la convención `tipo/descripcion-corta`.
- [ ] Rutas, fechas y semillas nuevas están solo en `mibici/config.py`.
- [ ] No se agregan datos, modelos, carpetas de MLflow (`mlruns/`, `mlartifacts/`) ni credenciales.
- [ ] No se registran datos personales en params, tags o artefactos de MLflow.
- [ ] Las dependencias nuevas están en `requirements.txt` con versión fija (`==`).
- [ ] `make lint` y `make test` pasan.
- [ ] Se actualizó el README si cambia la instalación o los comandos.
