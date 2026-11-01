# API operations

The FastAPI service is read-only and listens on `127.0.0.1:8000` by default:

```sh
python -m quantagent.cli api serve
```

Use `--host` and `--port` to change the bind address.

The API reads the account registry from the database selected by `QUANTAGENT_DATABASE_URL`. This is the master accounts database. Each paper portfolio keeps its own SQLite database and Alpaca account; the master database only registers portfolios with `api accounts add --database-url`. The web app reads portfolio data and does not process it.

The API has no application authentication of its own. Keep it bound to localhost and put an authenticating proxy in front of it before exposing it beyond the host. The same process can serve the React web app and the API.

## Register accounts

Register each account in the master database with `api accounts add`. The account's `--database-url` must point to an existing readable SQLite database:

```sh
python -m quantagent.cli api accounts add --id paper-nucleo-curva --name 'Paper Nucleo Curva' --env paper --kind alpaca --database-url sqlite:////srv/quantagent/paper.db --strategy nucleo-curva
```

Use `python -m quantagent.cli api accounts list` to inspect active accounts. The `api accounts show` and `api accounts archive` commands inspect or hide a configured account.

## OpenAPI contract

The frozen contract is [`docs/api/openapi.json`](../api/openapi.json). Regenerate it from the running application definition with:

```sh
python -m quantagent.cli api schema
```

Verify that the frozen file matches the live schema without writing it:

```sh
python -m quantagent.cli api schema --check
```

Use `--output PATH` to write or check a different JSON file.

## Paper database migration

`api migrate-paper-db --database-url URL` reports missing tables and columns without changing the database. Add `--apply` to create missing tables and safe columns, preserve existing rows, and stamp Alembic at head. Columns that are required without a default stop the migration. Before the API reads a paper database, run this command for that database: inspect the dry run first, then use `--apply`.

## Web app

Build the React app with `cd web && npm ci && npm run build`. Set `QUANTAGENT_WEB_DIST` to serve it from the API root; the default build directory is `web/dist`. Deep links fall back to the app's `index.html`, while `/api/*` remains routed to the API.

## Endpoints

All endpoints use `GET`. Response bodies are JSON; `422` responses use FastAPI's validation error schema.

| Method | Path | Responses | Parameters |
|---|---|---|---|
| GET | `/health` | `200` JSON object with `status: string` | None |
| GET | `/api/calendar` | `200` `CalendarEventResponse[]`; `422` validation error | `days` query, integer, default `30`, range `1`–`120` |
| GET | `/api/accounts` | `200` `AccountRef[]` | None |
| GET | `/api/accounts/{account_id}/summary` | `200` `AccountSummaryActive` or `AccountSummaryNoData`; `422` validation error | `account_id` path, string, required |
| GET | `/api/accounts/{account_id}/equity` | `200` `EquityPoint[]`; `404` unknown/inactive account; `422` validation error | `account_id` path, string, required; `range` query, `1W`, `1M`, `3M`, `YTD`, or `ALL`, default `1M` |
| GET | `/api/accounts/{account_id}/dividends` | `200` `DividendResponse[]`; `404` unknown/inactive account | `account_id` path, string, required |
| GET | `/api/accounts/{account_id}/sector-exposure` | `200` `SectorExposure[]`; `404` unknown/inactive account | `account_id` path, string, required |
| GET | `/api/accounts/{account_id}/positions` | `200` `PositionResponse[]`; `422` validation error | `account_id` path, string, required |
| GET | `/api/accounts/{account_id}/orders` | `200` `OrderResponse[]`; `422` validation error | `account_id` path, string, required; `date` query, `YYYY-MM-DD`, optional (defaults to today in New York) |

`GET /api/accounts/{account_id}/equity` reads history from daily equity snapshots (`equity`, `cash`, and `invested`) in that portfolio's database. It includes optional SPY and shadow values when the configured snapshot or shadow file has matching dates. Dividends come from account-filtered master database events. Sector exposure reads portfolio positions, groups GICS sectors (or unclassified holdings), and reports cash from the latest equity snapshot.
