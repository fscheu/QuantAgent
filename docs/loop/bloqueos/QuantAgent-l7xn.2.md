# QuantAgent-l7xn.2 — overflow existente en Cuenta

QA-02 no puede entregarse verde sin modificar pantallas, algo excluido por el ticket.

## Intento y resultado

- Se probaron @playwright/test 1.60.0, vite preview y siete rutas a 390/1280px.
- Esa versión usa Chromium revision1223, ya instalado: no descargó navegadores.
- Se reutilizaron los handlers y datos de web/src/mocks desde msw/node, con
  page.route para que las respuestas estuvieran listas antes de cargar React.
- `cd web && npm test`: 70 passed, exit 0.
- `cd web && npm run e2e`: 12 passed, 2 failed, exit 1.
- `/cuenta` y `/cuenta/paper-m4`, a 1280px: document.documentElement.scrollWidth
  devuelve 1340; se esperaba <=1280. Las dos capturas muestran la tabla de
  posiciones y el resto de Cuenta cargados con los fixtures existentes.
- Los otros doce casos pasaron. Se generaron los catorce PNG.
- Una inyección temporal de un div de 2000px hizo fallar el caso de 390px
  con scrollWidth=2008, confirmando que la aserción detecta overflow.
- `scripts/loop/smoke.sh`: exit 0, Trades 9, Total PnL 561.18.

## Reproducción y decisión

Con la web levantada (`cd web && npm ci && npm run dev -- --host 127.0.0.1`),
abrir `/cuenta/paper-m4?mock=1`, viewport de 1280px, esperar las posiciones
SPY/TSLA y evaluar `document.documentElement.scrollWidth` y `innerWidth`.
El arranque del worker MSW actual tiene una carrera: si aparece error de API,
recargar hasta ver las posiciones; el spec ensayado evitó esa carrera con MSW Node.

Destraba: crear un ticket previo que corrija el overflow de Cuenta a 1280px;
después retomar QA-02. No ocultar el overflow ni reducir los datos del fixture.
No se entregó código porque el contrato exige tests verdes y prohíbe cambios de
pantallas. El aislamiento de Codex impide publicar loop-wip: .git es de solo
lectura; el wrapper publica esta entrega de bloqueo y QA-02 sigue in_progress.
