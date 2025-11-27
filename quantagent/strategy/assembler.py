"""Strategy Assembler (Factory)

Builds a consistent set of trading components from a resolved configuration
snapshot: PortfolioManager, PositionSizer, RiskManager, Broker, OrderManager,
and TradingGraph, plus utilities to manage universe and config snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from quantagent.models import Environment
from quantagent.portfolio.manager import PortfolioManager
from quantagent.trading.position_sizer import PositionSizer
from quantagent.trading.risk_manager import RiskManager
from quantagent.trading.paper_broker import PaperBroker
from quantagent.trading.order_manager import OrderManager
from quantagent.trading_graph import TradingGraph


@dataclass(frozen=True)
class ResolvedConfig:
    environment: Environment
    universe: List[str]
    initial_cash: float
    base_position_pct: float
    max_daily_loss_pct: float
    max_position_pct: float
    slippage_pct: float
    model_provider: str
    model_name: str
    temperature: float
    use_checkpointing: bool
    extras: Optional[Dict] = None


@dataclass
class TradingComponents:
    portfolio_manager: PortfolioManager
    position_sizer: PositionSizer
    risk_manager: RiskManager
    broker: PaperBroker
    order_manager: OrderManager
    graph: TradingGraph


class StrategyAssembler:
    """Factory to resolve strategy config and assemble trading components."""

    DEFAULTS = {
        "initial_cash": 100000.0,
        "base_position_pct": 0.05,
        "max_daily_loss_pct": 0.05,
        "max_position_pct": 0.10,
        "slippage_pct": 0.01,
        "model_provider": "openai",
        "model_name": "gpt-4o-mini",
        "temperature": 0.1,
        "use_checkpointing": False,
        "universe": [],
    }

    @staticmethod
    def from_profiles(
        *,
        portfolio_profile: Optional[Dict] = None,
        risk_profile: Optional[Dict] = None,
        model_profile: Optional[Dict] = None,
        overrides: Optional[Dict] = None,
        environment: Environment = Environment.BACKTEST,
    ) -> ResolvedConfig:
        """Resolve configuration from profile dicts + overrides.

        Priority: overrides > model/portfolio/risk > defaults
        """
        portfolio_profile = portfolio_profile or {}
        risk_profile = risk_profile or {}
        model_profile = model_profile or {}
        overrides = overrides or {}

        merged: Dict = {**StrategyAssembler.DEFAULTS}

        # Portfolio fields
        merged.update({
            k: v for k, v in portfolio_profile.items()
            if k in {"universe", "base_position_pct", "slippage_pct", "initial_cash"}
        })

        # Risk fields
        merged.update({
            k: v for k, v in risk_profile.items()
            if k in {"max_daily_loss_pct", "max_position_pct"}
        })

        # Model fields (normalize to generic names)
        model_norm = StrategyAssembler._normalize_model_profile(model_profile)
        merged.update(model_norm)

        # Overrides last
        merged.update(overrides)

        # Build ResolvedConfig
        return StrategyAssembler._to_resolved(merged, environment)

    @staticmethod
    def from_snapshot(
        snapshot: Dict,
        *,
        environment: Environment = Environment.BACKTEST,
    ) -> ResolvedConfig:
        """Resolve configuration from a previously persisted snapshot dict.

        Accepts both the assembler schema (model_provider/model_name/temperature)
        and Backtest snapshot keys (agent_llm_*).
        """
        merged = {**StrategyAssembler.DEFAULTS, **snapshot}

        # Normalize model keys if they come from Backtest snapshot
        if "agent_llm_provider" in merged:
            merged.setdefault("model_provider", merged.get("agent_llm_provider"))
        if "agent_llm_model" in merged:
            merged.setdefault("model_name", merged.get("agent_llm_model"))
        if "agent_llm_temperature" in merged:
            merged.setdefault("temperature", merged.get("agent_llm_temperature"))

        return StrategyAssembler._to_resolved(merged, environment)

    @staticmethod
    def resolve_universe(resolved: ResolvedConfig, assets_override: Optional[List[str]]) -> List[str]:
        """Return assets_override if provided, else resolved.universe."""
        if assets_override is not None and len(assets_override) > 0:
            return assets_override
        return list(resolved.universe)

    @staticmethod
    def build_components(
        resolved: ResolvedConfig,
        *,
        db_session: Session,
    ) -> TradingComponents:
        """Construct trading components wired with the resolved config."""
        pm = PortfolioManager(
            initial_cash=resolved.initial_cash,
            environment=resolved.environment,
            db=db_session,
        )
        ps = PositionSizer(base_position_pct=resolved.base_position_pct)
        rm = RiskManager(
            portfolio_manager=pm,
            max_daily_loss_pct=resolved.max_daily_loss_pct,
            max_position_pct=resolved.max_position_pct,
            db=db_session,
        )
        broker = PaperBroker(slippage_pct=resolved.slippage_pct)
        om = OrderManager(
            position_sizer=ps,
            risk_manager=rm,
            broker=broker,
            portfolio_manager=pm,
            db=db_session,
        )

        # TradingGraph configuration mapping
        graph_cfg = {
            "agent_llm_provider": resolved.model_provider,
            "agent_llm_model": resolved.model_name,
            "agent_llm_temperature": resolved.temperature,
        }

        graph = TradingGraph(config=graph_cfg, use_checkpointing=resolved.use_checkpointing)

        return TradingComponents(
            portfolio_manager=pm,
            position_sizer=ps,
            risk_manager=rm,
            broker=broker,
            order_manager=om,
            graph=graph,
        )

    @staticmethod
    def config_snapshot(resolved: ResolvedConfig) -> Dict:
        """Create a serializable snapshot for persistence (BacktestRun)."""
        return {
            "initial_capital": resolved.initial_cash,
            "base_position_pct": resolved.base_position_pct,
            "max_daily_loss_pct": resolved.max_daily_loss_pct,
            "max_position_pct": resolved.max_position_pct,
            "slippage_pct": resolved.slippage_pct,
            # Align with TradingGraph expected keys (agent_llm_*)
            "agent_llm_provider": resolved.model_provider,
            "agent_llm_model": resolved.model_name,
            "agent_llm_temperature": resolved.temperature,
            "use_checkpointing": resolved.use_checkpointing,
            # Include universe for reproducibility (even if caller overrides)
            "universe": list(resolved.universe),
        }

    @staticmethod
    def make_thread_id(run_id: int, symbol: str, ts: datetime) -> str:
        return f"backtest_{run_id}_{symbol}_{ts.isoformat()}"

    # ---- Helpers ----
    @staticmethod
    def _normalize_model_profile(model_profile: Dict) -> Dict:
        out = {}
        # Accept both generic and agent_llm_* keys
        provider = model_profile.get("model_provider", model_profile.get("agent_llm_provider"))
        name = model_profile.get("model_name", model_profile.get("agent_llm_model"))
        temp = model_profile.get("temperature", model_profile.get("agent_llm_temperature"))

        if provider is not None:
            out["model_provider"] = provider
        if name is not None:
            out["model_name"] = name
        if temp is not None:
            out["temperature"] = temp

        # Runtime flags (non-secret)
        for k in ("use_checkpointing",):
            if k in model_profile:
                out[k] = model_profile[k]

        return out

    @staticmethod
    def _to_resolved(merged: Dict, environment: Environment) -> ResolvedConfig:
        # Basic validations and fallbacks
        universe = merged.get("universe") or []
        base_pct = float(merged.get("base_position_pct", StrategyAssembler.DEFAULTS["base_position_pct"]))
        max_pos = float(merged.get("max_position_pct", StrategyAssembler.DEFAULTS["max_position_pct"]))
        max_loss = float(merged.get("max_daily_loss_pct", StrategyAssembler.DEFAULTS["max_daily_loss_pct"]))
        slip = float(merged.get("slippage_pct", StrategyAssembler.DEFAULTS["slippage_pct"]))
        initial_cash = float(merged.get("initial_cash", StrategyAssembler.DEFAULTS["initial_cash"]))
        temp = float(merged.get("temperature", StrategyAssembler.DEFAULTS["temperature"]))
        use_ckpt = bool(merged.get("use_checkpointing", StrategyAssembler.DEFAULTS["use_checkpointing"]))

        return ResolvedConfig(
            environment=environment,
            universe=list(universe),
            initial_cash=initial_cash,
            base_position_pct=base_pct,
            max_daily_loss_pct=max_loss,
            max_position_pct=max_pos,
            slippage_pct=slip,
            model_provider=str(merged.get("model_provider", StrategyAssembler.DEFAULTS["model_provider"])),
            model_name=str(merged.get("model_name", StrategyAssembler.DEFAULTS["model_name"])),
            temperature=temp,
            use_checkpointing=use_ckpt,
            extras=None,
        )
