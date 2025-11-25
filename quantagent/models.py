"""SQLAlchemy models for QuantAgent trading system."""

from datetime import datetime
from decimal import Decimal
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Enum, ForeignKey,
    Text, Boolean, Numeric, Index
)
from sqlalchemy.orm import relationship
import enum

from .database import Base


class OrderStatus(str, enum.Enum):
    """Enum for order statuses."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderSide(str, enum.Enum):
    """Enum for order side (buy/sell)."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, enum.Enum):
    """Enum for order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TradeSignal(str, enum.Enum):
    """Enum for trading signals."""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"
    CLOSE = "close"


class Order(Base):
    """Order model for placing and tracking trading orders."""

    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(Enum(OrderSide), nullable=False)
    order_type = Column(Enum(OrderType), nullable=False)
    quantity = Column(Numeric(precision=18, scale=8), nullable=False)
    price = Column(Numeric(precision=18, scale=8), nullable=True)
    stop_price = Column(Numeric(precision=18, scale=8), nullable=True)
    status = Column(Enum(OrderStatus), nullable=False, default=OrderStatus.PENDING)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    filled_quantity = Column(Numeric(precision=18, scale=8), nullable=False, default=0)
    average_fill_price = Column(Numeric(precision=18, scale=8), nullable=True)
    comment = Column(Text, nullable=True)

    # Relationships
    fills = relationship("Fill", back_populates="order", cascade="all, delete-orphan")
    trades = relationship("Trade", back_populates="order", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_symbol_created_at", "symbol", "created_at"),
        Index("idx_status_symbol", "status", "symbol"),
    )


class Fill(Base):
    """Fill model for partial/full order executions."""

    __tablename__ = "fills"

    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(Integer, ForeignKey("orders.id"), nullable=False, index=True)
    quantity = Column(Numeric(precision=18, scale=8), nullable=False)
    price = Column(Numeric(precision=18, scale=8), nullable=False)
    commission = Column(Numeric(precision=18, scale=8), nullable=False, default=0)
    filled_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    # Relationships
    order = relationship("Order", back_populates="fills")

    __table_args__ = (
        Index("idx_order_filled_at", "order_id", "filled_at"),
    )


class Position(Base):
    """Position model for tracking open positions."""

    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, unique=True, index=True)
    quantity = Column(Numeric(precision=18, scale=8), nullable=False)
    average_entry_price = Column(Numeric(precision=18, scale=8), nullable=False)
    current_price = Column(Numeric(precision=18, scale=8), nullable=False)
    unrealized_pnl = Column(Numeric(precision=18, scale=8), nullable=False)
    unrealized_pnl_pct = Column(Float, nullable=False)
    side = Column(Enum(OrderSide), nullable=False)
    opened_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class Signal(Base):
    """Signal model for tracking trading signals from agents."""

    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    signal = Column(Enum(TradeSignal), nullable=False)
    confidence = Column(Float, nullable=False)  # 0-1 scale
    timeframe = Column(String(10), nullable=False)  # 1m, 5m, 15m, 1h, 4h, 1d, etc.

    # Agent analysis details
    rsi = Column(Float, nullable=True)
    macd = Column(Float, nullable=True)
    stochastic = Column(Float, nullable=True)
    roc = Column(Float, nullable=True)
    williams_r = Column(Float, nullable=True)
    pattern = Column(String(100), nullable=True)
    trend = Column(String(50), nullable=True)

    # Analysis report
    analysis_summary = Column(Text, nullable=True)
    generated_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    __table_args__ = (
        Index("idx_symbol_generated_at", "symbol", "generated_at"),
        Index("idx_symbol_signal", "symbol", "signal"),
    )


class Trade(Base):
    """Trade model for tracking completed trades."""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    order_id = Column(Integer, ForeignKey("orders.id"), nullable=True)
    entry_price = Column(Numeric(precision=18, scale=8), nullable=False)
    exit_price = Column(Numeric(precision=18, scale=8), nullable=True)
    quantity = Column(Numeric(precision=18, scale=8), nullable=False)
    side = Column(Enum(OrderSide), nullable=False)
    pnl = Column(Numeric(precision=18, scale=8), nullable=True)
    pnl_pct = Column(Float, nullable=True)
    commission = Column(Numeric(precision=18, scale=8), nullable=False, default=0)

    entry_signal = Column(String(50), nullable=True)
    exit_signal = Column(String(50), nullable=True)
    timeframe = Column(String(10), nullable=True)

    opened_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    closed_at = Column(DateTime, nullable=True)

    notes = Column(Text, nullable=True)

    # Relationships
    order = relationship("Order", back_populates="trades")

    __table_args__ = (
        Index("idx_symbol_opened_at", "symbol", "opened_at"),
        Index("idx_symbol_closed_at", "symbol", "closed_at"),
    )


class MarketData(Base):
    """MarketData model for storing historical OHLCV data."""

    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)  # 1m, 5m, 15m, 1h, 4h, 1d, etc.
    timestamp = Column(DateTime, nullable=False, index=True)

    open = Column(Numeric(precision=18, scale=8), nullable=False)
    high = Column(Numeric(precision=18, scale=8), nullable=False)
    low = Column(Numeric(precision=18, scale=8), nullable=False)
    close = Column(Numeric(precision=18, scale=8), nullable=False)
    volume = Column(Numeric(precision=20, scale=8), nullable=False)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_symbol_timeframe_timestamp", "symbol", "timeframe", "timestamp"),
        Index("idx_timestamp", "timestamp"),
    )
