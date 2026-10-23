# Mapa de lotes

El workflow [Mapa de lotes](../../.github/workflows/mapa-lotes.yml) genera el HTML y lo publica en GitHub Pages al actualizar `main` o `lote/**`, y una vez por día.

Para actualizar el mapa, editá `lotes.json` (lotes y asignaciones) o `sessions.json` (sesiones activas y planificadas) y subí los cambios al repositorio. GitHub Pages debe estar habilitado en Settings → Pages, con **Source: GitHub Actions**.
