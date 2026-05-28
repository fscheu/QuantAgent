from __future__ import annotations

import json
from typing import Dict, List

import pandas as pd
import streamlit as st

from quantagent.data.provider import DataProvider
from quantagent.llm.registry import supported_providers
from quantagent.llm.roles import ProviderRoleConfig
from quantagent.llm.routing import ProviderRoutingPolicy
from quantagent.strategy.registry import get_strategy_names

SUPPORTED_UNIVERSE_SYMBOLS: List[str] = list(DataProvider.SYMBOL_MAPPING.keys())


def _collect_profiles_from_db(db, kind: str) -> List[str]:
    if not db.ok:
        return []
    try:
        with db.SessionLocal() as s:
            return [
                c.name
                for c in s.query(db.models.StrategyConfig)
                .filter_by(kind=kind)
                .order_by(db.models.StrategyConfig.name)
                .all()
            ]
    except Exception:
        return []


def _get_profile_json_from_db(db, name: str):
    if not db.ok:
        return None
    try:
        with db.SessionLocal() as s:
            cfg = s.query(db.models.StrategyConfig).filter_by(name=name).one_or_none()
            return cfg.json_config if cfg else None
    except Exception:
        return None


def render(db, environment: str) -> None:
    st.subheader("Configuration – Strategy Profiles & Model Presets")
    st.caption(
        "Profiles are persisted to the database when available; session fallback remains for offline use."
    )

    st.session_state.setdefault(
        "ui_profiles", {"portfolio": {}, "risk": {}, "combined": {}}
    )
    st.session_state.setdefault(
        "model_presets",
        {
            "default": {
                "provider": "openai",
                "model_name": "gpt-4o-mini",
                "temperature": 0.1,
            }
        },
    )
    st.session_state.setdefault(
        "provider_routing_presets",
        {
            "default": {
                "deep_reasoning": {
                    "provider": "openai",
                    "model_name": "gpt-4o",
                    "temperature": 0.1,
                },
                "lite": {
                    "provider": "openai",
                    "model_name": "gpt-4o-mini",
                    "temperature": 0.1,
                },
                "image": None,
            }
        },
    )
    st.session_state.setdefault("default_profiles", {"paper": None, "backtest": None})
    st.session_state.setdefault("default_strategy", {"paper": None, "backtest": None})

    # Profile editor
    colL, colR = st.columns([2, 1])
    with colL:
        kind = st.selectbox("Profile kind", ["portfolio", "risk", "combined"], index=2)
        name = st.text_input("Profile name", value="default")

        existing_json = _get_profile_json_from_db(
            db, name
        ) or st.session_state.ui_profiles.get(kind, {}).get(name)
        raw_default = json.dumps(
            existing_json
            or {
                "universe": ["BTC", "SPX"],
                "base_position_pct": 0.05,
                "max_position_pct": 0.1,
                "max_daily_loss_pct": 0.05,
            },
            indent=2,
        )
        raw = st.text_area(
            "Profile JSON", value=raw_default, height=260, key=f"profile_json_{kind}"
        )

        universe_default: List[str] = []
        parsed = None
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                universe_default = parsed.get("universe", []) or []
        except Exception:
            parsed = None
        allowed_universe_default = [
            u for u in universe_default if u in SUPPORTED_UNIVERSE_SYMBOLS
        ]
        unsupported_universe = [
            u for u in universe_default if u not in SUPPORTED_UNIVERSE_SYMBOLS
        ]

        universe: List[str] = allowed_universe_default
        if kind == "portfolio":
            universe = st.multiselect(
                "Universe (portfolio profiles only)",
                SUPPORTED_UNIVERSE_SYMBOLS,
                default=allowed_universe_default,
            )
        else:
            st.caption("Universe editing is available for portfolio profiles only.")

        resolved_universe = (
            universe if kind == "portfolio" else allowed_universe_default
        )
        st.markdown("**Universe preview**")
        if resolved_universe:
            st.dataframe(
                pd.DataFrame({"symbol": resolved_universe}),
                width='stretch',
            )
        else:
            st.info("No symbols selected for this profile.")
        if unsupported_universe:
            st.warning(
                "Unsupported symbols were ignored: "
                + ", ".join(sorted(set(unsupported_universe)))
            )

        if st.button("Save profile"):
            try:
                data = json.loads(raw)
                if kind == "portfolio":
                    data["universe"] = universe
                if db.ok:
                    with db.SessionLocal() as s:
                        existing = (
                            s.query(db.models.StrategyConfig)
                            .filter_by(name=name)
                            .one_or_none()
                        )
                        if existing:
                            existing.kind = kind
                            existing.json_config = data
                            existing.version = (existing.version or 1) + 1
                        else:
                            s.add(
                                db.models.StrategyConfig(
                                    name=name, kind=kind, json_config=data
                                )
                            )
                        s.commit()
                    st.success(f"Saved {kind} profile '{name}' to database.")
                else:
                    st.session_state.ui_profiles.setdefault(kind, {})[name] = data
                    st.success(f"Saved {kind} profile '{name}' to session.")
            except Exception as e:
                st.error(f"Invalid JSON: {e}")

        st.markdown("**Profiles**")
        profiles_rows: List[Dict] = []
        if db.ok:
            with db.SessionLocal() as s:
                try:
                    for cfg in (
                        s.query(db.models.StrategyConfig)
                        .order_by(db.models.StrategyConfig.created_at.desc())
                        .all()
                    ):
                        profiles_rows.append(
                            {
                                "source": "db",
                                "kind": cfg.kind,
                                "name": cfg.name,
                                "version": cfg.version,
                                "updated_at": cfg.updated_at,
                            }
                        )
                except Exception as e:  # pragma: no cover - display only
                    st.info(f"Could not load DB profiles: {e}")
        for k, d in st.session_state.ui_profiles.items():
            for n, v in d.items():
                profiles_rows.append(
                    {
                        "source": "session",
                        "kind": k,
                        "name": n,
                        "version": "-",
                        "updated_at": "-",
                    }
                )
        st.dataframe(pd.DataFrame(profiles_rows), width='stretch')

    with colR:
        st.markdown("**Defaults per environment**")
        portfolio_names = _collect_profiles_from_db(db, "portfolio") or list(
            st.session_state.ui_profiles.get("portfolio", {}).keys()
        )
        for env_key in ("paper", "backtest"):
            options = ["(none)"] + portfolio_names
            current_default = st.session_state.default_profiles.get(env_key) or "(none)"
            chosen = st.selectbox(
                f"{env_key.title()} default portfolio",
                options,
                index=(
                    options.index(current_default) if current_default in options else 0
                ),
                key=f"default_{env_key}",
            )
            if st.button(f"Set {env_key} default", key=f"btn_default_{env_key}"):
                st.session_state.default_profiles[env_key] = (
                    None if chosen == "(none)" else chosen
                )
                st.success(
                    f"Default for {env_key} set to {st.session_state.default_profiles[env_key]}"
                )

        st.markdown("**Strategy Defaults**")
        strategy_names = ["(none)"] + get_strategy_names()
        for env_key in ("paper", "backtest"):
            current_strategy = st.session_state.default_strategy.get(env_key) or "(none)"
            chosen_strategy = st.selectbox(
                f"{env_key.title()} default strategy",
                strategy_names,
                index=(
                    strategy_names.index(current_strategy)
                    if current_strategy in strategy_names
                    else 0
                ),
                key=f"default_strategy_{env_key}",
            )
            if st.button(
                f"Set {env_key} strategy default",
                key=f"btn_default_strategy_{env_key}",
            ):
                st.session_state.default_strategy[env_key] = (
                    None if chosen_strategy == "(none)" else chosen_strategy
                )
                st.success(
                    "Strategy default for "
                    f"{env_key} set to {st.session_state.default_strategy[env_key]}"
                )

        st.markdown("**Model presets**")
        preset_names = list(st.session_state.model_presets.keys())
        preset_name = st.selectbox(
            "Preset name",
            preset_names,
            index=preset_names.index("default") if "default" in preset_names else 0,
        )
        preset = st.session_state.model_presets.get(preset_name, {})
        provider_options = supported_providers()
        provider_default = preset.get("provider", "openai")
        provider_index = (
            provider_options.index(provider_default)
            if provider_default in provider_options
            else 0
        )
        provider = st.selectbox(
            "Provider", provider_options, index=provider_index, key="model_provider"
        )
        model_name = st.text_input(
            "Model name",
            value=preset.get("model_name", "gpt-4o-mini"),
            key="model_name",
        )
        temperature = st.slider(
            "Temperature",
            0.0,
            1.0,
            float(preset.get("temperature", 0.1)),
            key="model_temp",
        )
        new_name = st.text_input(
            "Save as (name)", value=preset_name, key="preset_new_name"
        )

        if st.button("Save preset"):
            st.session_state.model_presets[new_name] = {
                "provider": provider,
                "model_name": model_name,
                "temperature": temperature,
            }
            st.success(f"Saved preset '{new_name}'.")

        st.markdown("**Presets preview**")
        st.dataframe(
            pd.DataFrame.from_dict(st.session_state.model_presets, orient="index"),
            width='stretch',
        )

        st.markdown("**Provider routing presets**")
        routing_preset_names = _collect_profiles_from_db(db, "provider_routing") or list(
            st.session_state.provider_routing_presets.keys()
        )
        routing_preset_name = st.selectbox(
            "Routing preset",
            routing_preset_names,
            index=(
                routing_preset_names.index("default")
                if "default" in routing_preset_names
                else 0
            ),
            key="routing_preset_name",
        )
        routing_payload = _get_profile_json_from_db(db, routing_preset_name) or st.session_state.provider_routing_presets.get(
            routing_preset_name, {}
        )
        routing_policy = ProviderRoutingPolicy.from_dict(routing_payload or {})

        resolved_roles = {}
        for role_name, role_cfg in {
            "deep_reasoning": routing_policy.deep_reasoning,
            "lite": routing_policy.lite,
            "image": routing_policy.image,
        }.items():
            st.caption(role_name.replace("_", " ").title())
            resolved_roles[role_name] = ProviderRoleConfig(
                provider=st.selectbox(
                    f"Provider ({role_name})",
                    provider_options,
                    index=(
                        provider_options.index(role_cfg.provider)
                        if role_cfg and role_cfg.provider in provider_options
                        else 0
                    ),
                    key=f"routing_provider_{role_name}",
                ),
                model_name=st.text_input(
                    f"Model name ({role_name})",
                    value=(role_cfg.model_name if role_cfg else ""),
                    key=f"routing_model_{role_name}",
                ),
                temperature=st.slider(
                    f"Temperature ({role_name})",
                    0.0,
                    1.0,
                    float(role_cfg.temperature if role_cfg else 0.1),
                    key=f"routing_temp_{role_name}",
                ),
            )

        routing_save_name = st.text_input(
            "Routing save as (name)",
            value=routing_preset_name,
            key="routing_preset_save_name",
        )
        if st.button("Save routing preset", key="save_provider_routing_preset"):
            payload = ProviderRoutingPolicy(
                deep_reasoning=resolved_roles["deep_reasoning"],
                lite=resolved_roles["lite"],
                image=resolved_roles["image"] if resolved_roles["image"].model_name else None,
            ).to_dict()
            if db.ok:
                with db.SessionLocal() as s:
                    existing = (
                        s.query(db.models.StrategyConfig)
                        .filter_by(name=routing_save_name)
                        .one_or_none()
                    )
                    if existing:
                        existing.kind = "provider_routing"
                        existing.json_config = payload
                        existing.version = (existing.version or 1) + 1
                    else:
                        s.add(
                            db.models.StrategyConfig(
                                name=routing_save_name,
                                kind="provider_routing",
                                json_config=payload,
                            )
                        )
                    s.commit()
                st.success(f"Saved routing preset '{routing_save_name}' to database.")
            else:
                st.session_state.provider_routing_presets[routing_save_name] = payload
                st.success(f"Saved routing preset '{routing_save_name}' to session.")

        preview_policy = ProviderRoutingPolicy(
            deep_reasoning=resolved_roles["deep_reasoning"],
            lite=resolved_roles["lite"],
            image=resolved_roles["image"] if resolved_roles["image"].model_name else None,
        )
        preview_rows = []
        for role_name in ("deep_reasoning", "lite", "image"):
            try:
                role_cfg = preview_policy.resolve(role_name)
            except Exception:
                continue
            row = {
                "role": role_name,
                "provider": role_cfg.provider,
                "model_name": role_cfg.model_name,
                "temperature": role_cfg.temperature,
            }
            if role_name == "image" and preview_policy.image is None:
                row["resolved_from"] = "deep_reasoning"
            preview_rows.append(row)
        if preview_rows:
            st.dataframe(pd.DataFrame(preview_rows), width='stretch')
