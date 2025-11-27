from __future__ import annotations

import json

import pandas as pd
import streamlit as st


def render() -> None:
    st.subheader("Configuration – Strategy Profiles & Model Presets")
    st.caption("Profiles are kept in session for MVP until DB tables exist.")

    # Ensure session keys exist
    st.session_state.setdefault("ui_profiles", {"portfolio": {}, "risk": {}, "combined": {}})
    st.session_state.setdefault(
        "model_presets", {"default": {"provider": "openai", "model_name": "gpt-4o-mini", "temperature": 0.1}}
    )

    colL, colR = st.columns([2, 1])
    with colL:
        kind = st.selectbox("Profile kind", ["portfolio", "risk", "combined"], index=2)
        name = st.text_input("Profile name", value="default")
        raw = st.text_area("Profile JSON", value=json.dumps({"example": "value"}, indent=2), height=220)
        if st.button("Save profile"):
            try:
                data = json.loads(raw)
                st.session_state.ui_profiles.setdefault(kind, {})[name] = data
                st.success(f"Saved {kind} profile '{name}' to session.")
            except Exception as e:
                st.error(f"Invalid JSON: {e}")

        st.markdown("**Profiles**")
        profiles_df = []
        for k, d in st.session_state.ui_profiles.items():
            for n, v in d.items():
                profiles_df.append({"kind": k, "name": n, "size": len(json.dumps(v))})
        st.dataframe(pd.DataFrame(profiles_df), use_container_width=True)

    with colR:
        st.markdown("**Model preset**")
        default = st.session_state.model_presets.get("default", {})
        provider = st.selectbox("Provider", ["openai", "anthropic", "qwen"], index=0)
        model_name = st.text_input("Model name", value=default.get("model_name", "gpt-4o-mini"))
        temperature = st.slider("Temperature", 0.0, 1.0, float(default.get("temperature", 0.1)))
        if st.button("Save as default preset"):
            st.session_state.model_presets["default"] = {
                "provider": provider,
                "model_name": model_name,
                "temperature": temperature,
            }
            st.success("Saved default model preset.")

