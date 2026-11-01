# Catálogo de datos

Cada `datos/<id>.yaml` describe una familia de datos locales, su fuente, campos y lectores.
Los datos y manifiestos locales quedan fuera de git; `snapshots` lista sólo manifiestos JSON versionados.
Para agregar un dataset, creá su YAML siguiendo los existentes y registrá sus patrones `carpeta`.
`datos_esperados: true` exige que cada patrón tenga una carpeta cuando hay datos locales disponibles.
Usá `false` sólo para inventarios documentados ausentes y explicá esa ausencia en `descripcion`.
Los patrones son nombres directamente bajo `local-data/`; `_no-son-datos.yaml` excluye artefactos con motivo.
Verificá con `python -m pytest tests/test_catalogo_datos.py -q`.
Para otro checkout, anteponé `QUANTAGENT_SNAPSHOT_DIR=/ruta/local-data` al comando.
