# Partición de QuantAgent-dpr3.2

El ticket abarca 128 declaraciones `help=`/`epilog` en diez módulos de CLI. Al contar cada línea reemplazada como una baja y un alta, cada hijo queda debajo de 100 líneas:

- `QuantAgent-dpr3.2.1`: 46 declaraciones de `backtest.py` (92 líneas de diff estimadas).
- `QuantAgent-dpr3.2.2`: 40 declaraciones de `hipotesis.py`, `data.py` e `intentos.py` (80 líneas estimadas).
- `QuantAgent-dpr3.2.3`: 12 declaraciones de `reconcile.py`, `walkforward.py`, `verdict.py` y `__main__.py` (24 líneas), más el test global de Click (estimado en 25 líneas; 49 en total).
- `QuantAgent-dpr3.2.4`: 30 declaraciones de `sweep.py` y `profile.py` (60 líneas estimadas).

Cada hijo agrega help donde falte dentro de sus módulos. El test global final verifica help no vacío en todos los grupos, comandos y opciones y ausencia de `á é í ó ú ñ ¿ ¡`.
