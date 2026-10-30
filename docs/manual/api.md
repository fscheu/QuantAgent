# API operations

The FastAPI service is read-only and listens on `127.0.0.1:8000` by default. On the VM, start it with:

```sh
python -m quantagent.cli api serve
```

The API reads the account registry from the database selected by `QUANTAGENT_DATABASE_URL`. Set this environment variable to the master accounts database before running account-management commands or serving the API. The API itself has no application authentication. Access is private through the VM's existing Tailscale Serve setup; this guide does not enable or change that setup.

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

## Endpoints

All endpoints use `GET`. Response bodies are JSON; `422` responses use FastAPI's validation error schema.

| Method | Path | Responses | Parameters |
|---|---|---|---|
| GET | `/health` | `200` JSON object with `status: string` | None |
| GET | `/api/calendar` | `200` `CalendarEventResponse[]`; `422` validation error | `days` query, integer, default `30`, range `1`–`120` |
| GET | `/api/accounts` | `200` `AccountRef[]` | None |
| GET | `/api/accounts/{account_id}/summary` | `200` `AccountSummaryActive` or `AccountSummaryNoData`; `422` validation error | `account_id` path, string, required |
| GET | `/api/accounts/{account_id}/equity` | `200` `EquityPoint[]`; `422` validation error | `account_id` path, string, required; `range` query, `1W`, `1M`, `3M`, `YTD`, or `ALL`, default `1M` |
| GET | `/api/accounts/{account_id}/positions` | `200` `PositionResponse[]`; `422` validation error | `account_id` path, string, required |
| GET | `/api/accounts/{account_id}/orders` | `200` `OrderResponse[]`; `422` validation error | `account_id` path, string, required; `date` query, `YYYY-MM-DD`, optional (defaults to today in New York) |
