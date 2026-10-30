# QuantAgent-wsxd — Despliegue de la web

**Issue:** QuantAgent-wsxd (WEB-17)  
**Estado:** propuesta; no se aplicó ningún paso.  
**Alcance:** API y frontend web en la VM de paper, detrás de Cloudflare Access.

## 1. Prerrequisitos

1. Confirmar que lote/M-web quedó mergeado en `main`. **[OK Fede]**
2. Crear un tag nuevo de versión sobre ese `main` y usarlo para el despliegue. El checkout estable es `/home/azureuser/paper-ops/QuantAgent`; el venv es `/home/azureuser/paper-ops/venv`. El código desplegado debe incluir `trading snapshot`; el tag actual `paper-2026-11-02` no tiene ese comando.
3. Confirmar que existen las dos bases paper y `/home/azureuser/web-ops/maestro.db`. Las cuentas ya están dadas de alta en el maestro.
4. Confirmar acceso de Fede al dashboard Cloudflare Zero Trust. Cloudflare Access debe proteger la API; nunca apuntar el hostname a la API sin Access.

## 2. Backup y despliegue del tag

Antes de cambiar el checkout, respaldar las bases mediante SQLite `.backup` (no copiar archivos SQLite activos):

```bash
BACKUP=/home/azureuser/paper-ops/backups/$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$BACKUP"
sqlite3 /home/azureuser/paper-ops/QuantAgent/local-data/cartera-nucleo-curva.db ".backup '$BACKUP/cartera-nucleo-curva.db'"
sqlite3 /home/azureuser/paper-ops/QuantAgent/local-data/cartera-transversal.db ".backup '$BACKUP/cartera-transversal.db'"
sqlite3 /home/azureuser/web-ops/maestro.db ".backup '$BACKUP/maestro.db'"
```

Desplegar el tag aprobado en el clon paper, conservando el tag anterior para rollback. **[OK Fede]**

```bash
cd /home/azureuser/paper-ops/QuantAgent
git fetch --tags
git checkout <TAG_NUEVO>
/home/azureuser/paper-ops/venv/bin/pip install -e .
cd web && npm ci && npm run build
```

El instalador [install.sh](../../deploy/web/install.sh) hace backup, build, migración con `--apply`, instala las unidades y habilita la API. Por defecto sólo imprime los pasos. Revisá primero con `DRY_RUN=1 bash deploy/web/install.sh`; ejecutalo con `DRY_RUN=0` sólo después de aprobar el despliegue. El instalador presupone que el código nuevo ya está en el checkout `paper-ops` y no cambia el tag.

## 3. Migración de bases

Primero inspeccionar el plan sin modificar ninguna base:

```bash
cd /home/azureuser/paper-ops/QuantAgent
for cartera in cartera-nucleo-curva cartera-transversal; do
  /home/azureuser/paper-ops/venv/bin/quantagent api migrate-paper-db \
    --database-url "sqlite:////home/azureuser/paper-ops/QuantAgent/local-data/${cartera}.db"
done
```

Revisar la salida de ambas. Aplicar únicamente si las migraciones reportadas son las esperadas. **[OK Fede]**

```bash
for cartera in cartera-nucleo-curva cartera-transversal; do
  /home/azureuser/paper-ops/venv/bin/quantagent api migrate-paper-db \
    --database-url "sqlite:////home/azureuser/paper-ops/QuantAgent/local-data/${cartera}.db" --apply
done
```

Ante una migración inesperada, detenerse y restaurar desde el backup antes de continuar.

## 4. Cuentas y API

Las cuentas ya están registradas en el maestro. Verificarlas ahí:

```bash
QUANTAGENT_DATABASE_URL=sqlite:////home/azureuser/web-ops/maestro.db \
  /home/azureuser/paper-ops/venv/bin/quantagent api accounts list
```

Si hiciera falta registrar una cuenta en una base paper, el patrón CLI es:

```bash
QUANTAGENT_DATABASE_URL=sqlite:////home/azureuser/web-ops/maestro.db \
  /home/azureuser/paper-ops/venv/bin/quantagent api accounts add \
  --id <cartera> --name '<nombre>' --env paper --kind alpaca \
  --database-url sqlite:////home/azureuser/paper-ops/QuantAgent/local-data/<cartera>.db \
  --strategy <estrategia>
```

Instalar `deploy/web/quantagent-api.service` en `~/.config/systemd/user/`, recargar systemd y arrancar la API. **[OK Fede]**

```bash
systemctl --user daemon-reload
systemctl --user enable --now quantagent-api.service
systemctl --user status quantagent-api.service
curl --fail http://127.0.0.1:8002/health
```

## 5. Snapshot junto al digest

Instalar `deploy/web/paper-digest-snapshot.conf` como `~/.config/systemd/user/paper-digest@.service.d/snapshot.conf` y ejecutar `systemctl --user daemon-reload`. **[OK Fede]** El drop-in hereda el `EnvironmentFile` existente de `paper-digest@.service`. El prefijo `-` de `ExecStartPost` hace que un error del snapshot no falle el digest ni afecte los timers. El comando requiere el tag nuevo: el tag anterior `paper-2026-11-02` no contiene `trading snapshot`.

No se detienen ni reinician los timers paper durante este despliegue. Verificar sus estados antes y después:

```bash
systemctl --user list-timers 'paper-cartera@*' 'paper-digest@*'
```

## 6. Cloudflare Access e ingress

Propuesta de hostname: `quantagent.fedes.dev`. El hostname definitivo es `quantagent.fedes.dev`. Cambiar la ruta actual corta Streamlit en esa URL. **[OK Fede]**

En Cloudflare Zero Trust, Fede debe:

1. Ir a **Access > Applications > Add an application > Self-hosted**. **[OK Fede]**
2. Crear la aplicación con el hostname `quantagent.fedes.dev`. **[OK Fede]**
3. Crear una policy **Allow** cuyo selector incluya el email de Fede. **[OK Fede]**
4. Elegir método de login **One-time PIN** o **Google** y guardar la aplicación/policy. **[OK Fede]**
5. En `/etc/cloudflared/config.yml`, reemplazar la regla `quantagent.fedes.dev -> http://localhost:8501` por el bloque de [cloudflared-ingress.yml.example](../../deploy/web/cloudflared-ingress.yml.example), sustituyendo `quantagent.fedes.dev` por el hostname confirmado. Preservar el resto de la configuración y el `ingress` final de captura. **[OK Fede]**
6. Mover o retirar la ruta vieja de Streamlit asociada a ese hostname; al reiniciar el tunnel, esa URL deja de servir Streamlit. **[OK Fede]**
7. Reiniciar el servicio del tunnel: `sudo systemctl restart cloudflared`. **[OK Fede]**

## 7. Verificación

Desde la VM, comprobar que la API responde localmente:

```bash
curl --fail http://127.0.0.1:8002/api/accounts
```

Desde fuera de una sesión Access, `https://quantagent.fedes.dev/api/accounts` debe quedar bloqueado o redirigir al login de Access. Tras completar el login autorizado de Fede, la misma URL debe responder JSON con las cuentas configuradas. Abrir `https://quantagent.fedes.dev/` en el navegador y comprobar que carga la web y sus datos.

## 8. Rollback

- Si falla el build, migración o arranque antes del cambio Cloudflare, volver al tag anterior (`paper-2026-11-02`), reinstalar dependencias de ese tag y reconstruir web. Si ya se aplicó una migración, restaurar las tres bases desde el backup correspondiente antes de reabrir la API.
- Si falla la API después de instalar la unidad, ejecutar `systemctl --user disable --now quantagent-api.service`, volver al tag anterior y restaurar las bases desde backup si se migraron.
- Si falla el cutover Cloudflare, restaurar el ingress anterior `quantagent.fedes.dev -> http://localhost:8501` y ejecutar `sudo systemctl restart cloudflared`. Esto devuelve Streamlit a esa URL. Revisar o retirar la aplicación Access según lo que Fede decida.
- Los timers `paper-cartera@` y `paper-digest@` no se detienen ni reinician como parte del despliegue o rollback.

## Ventana de aplicación

Aplicar fuera de las corridas paper (ciclo 08:00 UTC y digest 08:20 UTC, martes a sábado): sábado o domingo, antes de la primera rueda del 2026-11-02, y siempre después del backup de §Backup.
