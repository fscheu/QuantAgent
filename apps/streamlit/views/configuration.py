from __future__ import annotations

import json
from typing import Dict, List

import pandas as pd
import streamlit as st


def _collect_profiles_from_db(db, kind: str) -> List[str]:
    if not db.ok:
        return []
    try:
        with db.SessionLocal() as s:
            return [c.name for c in s.query(db.models.StrategyConfig).filter_by(kind=kind).order_by(db.models.StrategyConfig.name).all()]
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
    st.caption("Profiles are persisted to the database when available; session fallback remains for offline use.")

    st.session_state.setdefault("ui_profiles", {"portfolio": {}, "risk": {}, "combined": {}})
    st.session_state.setdefault(
        "model_presets", {"default": {"provider": "openai", "model_name": "gpt-4o-mini", "temperature": 0.1}}
    )
    st.session_state.setdefault("default_profiles", {"paper": None, "backtest": None})

    # Profile editor
    colL, colR = st.columns([2, 1])
    with colL:
        kind = st.selectbox("Profile kind", ["portfolio", "risk", "combined"], index=2)
        name = st.text_input("Profile name", value="default")

        existing_json = _get_profile_json_from_db(db, name) or st.session_state.ui_profiles.get(kind, {}).get(name)
        raw_default = json.dumps(
            existing_json
            or {"universe": ["BTC", "SPX"], "base_position_pct": 0.05, "max_position_pct": 0.1, "max_daily_loss_pct": 0.05},
            indent=2,
        )
        raw = st.text_area("Profile JSON", value=raw_default, height=260, key=f"profile_json_{kind}")

        universe_default: List[str] = []
        try:
            parsed = json.loads(raw)
            universe_default = parsed.get("universe", [])
        except Exception:
            pass
        universe = st.multiselect(
            "Universe (applied on save for portfolio profiles)", ["BTC", "ETH", "SPX", "NQ", "CL", "GC", "AAPL", "QQQ"],
            default=universe_default,
        )

        if st.button("Save profile"):
            try:
                data = json.loads(raw)
                if kind == "portfolio" and universe:
                    data["universe"] = universe
                if db.ok:
                    with db.SessionLocal() as s:
                        existing = s.query(db.models.StrategyConfig).filter_by(name=name).one_or_none()
                        if existing:
                            existing.kind = kind
                            existing.json_config = data
                            existing.version = (existing.version or 1) + 1
                        else:
                            s.add(db.models.StrategyConfig(name=name, kind=kind, json_config=data))
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
                    for cfg in s.query(db.models.StrategyConfig).order_by(db.models.StrategyConfig.created_at.desc()).all():
                        profiles_rows.append(
                            {"source": "db", "kind": cfg.kind, "name": cfg.name, "version": cfg.version, "updated_at": cfg.updated_at}
                        )
                except Exception as e:  # pragma: no cover - display only
                    st.info(f"Could not load DB profiles: {e}")
        for k, d in st.session_state.ui_profiles.items():
            for n, v in d.items():
                profiles_rows.append({"source": "session", "kind": k, "name": n, "version": "-", "updated_at": "-"})
        st.dataframe(pd.DataFrame(profiles_rows), use_container_width=True)

    with colR:
        st.markdown("**Defaults per environment**")
        portfolio_names = _collect_profiles_from_db(db, "portfolio") or list(st.session_state.ui_profiles.get("portfolio", {}).keys())
        for env_key in ("paper", "backtest"):
            options = ["(none)"] + portfolio_names
            current_default = st.session_state.default_profiles.get(env_key) or "(none)"
            chosen = st.selectbox(f"{env_key.title()} default portfolio", options, index=options.index(current_default) if current_default in options else 0, key=f"default_{env_key}")
            if st.button(f"Set {env_key} default", key=f"btn_default_{env_key}"):
                st.session_state.default_profiles[env_key] = None if chosen == "(none)" else chosen
                st.success(f"Default for {env_key} set to {st.session_state.default_profiles[env_key]}")

        st.markdown("**Model presets**")
        preset_names = list(st.session_state.model_presets.keys())
        preset_name = st.selectbox("Preset name", preset_names, index=preset_names.index("default") if "default" in preset_names else 0)
        preset = st.session_state.model_presets.get(preset_name, {})
        provider_options = ["openai", "anthropic", "qwen"]
        provider_default = preset.get("provider", "openai")
        provider_index = provider_options.index(provider_default) if provider_default in provider_options else 0
        provider = st.selectbox("Provider", provider_options, index=provider_index, key="model_provider")
        model_name = st.text_input("Model name", value=preset.get("model_name", "gpt-4o-mini"), key="model_name")
        temperature = st.slider("Temperature", 0.0, 1.0, float(preset.get("temperature", 0.1)), key="model_temp")
        new_name = st.text_input("Save as (name)", value=preset_name, key="preset_new_name")

        if st.button("Save preset"):
            st.session_state.model_presets[new_name] = {
                "provider": provider,
                "model_name": model_name,
                "temperature": temperature,
            }
            st.success(f"Saved preset '{new_name}'.")

        st.markdown("**Presets preview**")
        st.dataframe(pd.DataFrame.from_dict(st.session_state.model_presets, orient="index"), use_container_width=True)

