# Strategy Configuration Guide

Strategy configuration controls **what you trade** and **how much risk you take**. This guide explains how to create, save, and manage strategy profiles.

---

## Understanding Strategy Profiles

A strategy profile combines three key elements:

1. **Universe** - Which assets to trade (BTC, SPX, CL, etc.)
2. **Position Sizing** - How much capital to allocate per trade
3. **Risk Limits** - Maximum losses and position sizes

**Think of it like a recipe:** The profile defines the ingredients (assets) and proportions (sizing) for your trading strategy.

---

## Profile Types

QuantAgent supports three profile types:

### 1. Portfolio Profile

**Focuses on:** What to trade and how much

**Contains:**
- Universe (list of assets)
- Base position size percentage
- Maximum position size per asset

**Use when:** You want to manage position sizing separately from risk rules

### 2. Risk Profile

**Focuses on:** Loss limits and circuit breakers

**Contains:**
- Maximum daily loss before stopping
- Maximum open positions
- Slippage simulation percentage

**Use when:** Testing different risk levels on the same assets

### 3. Combined Profile (Recommended)

**Focuses on:** Everything in one place

**Contains:** All portfolio AND risk settings

**Use when:** Starting out or wanting simplicity

**Most users should use combined profiles.**

---

## Creating Profiles

**Location:** Dashboard → Configuration tab

### Step-by-Step: Create a Combined Profile

<!-- screenshot: Empty configuration form -->

#### 1. Select Profile Type

**Profile kind dropdown:** Choose `combined`

#### 2. Name Your Profile

**Profile name field:** Enter a descriptive name

**Good names:**
- `conservative-crypto` - Low risk, cryptocurrency focus
- `aggressive-multi-asset` - High risk, diversified
- `btc-4h-standard` - BTC only, 4-hour timeframe, moderate risk

**Avoid generic names like "profile1" or "test"** - you'll forget what they do.

#### 3. Choose Assets (Universe)

**Universe multi-select:** Click to select one or more assets

**Supported symbols:**
- **BTC** - Bitcoin
- **SPX** - S&P 500 Index
- **CL** - Crude Oil (WTI)
- **ETH** - Ethereum (if configured)
- *(Additional symbols depend on yfinance support)*

**Tips:**
- Start with 1-2 assets for testing
- Mix crypto + traditional (BTC + SPX) for diversification
- More assets = more potential trades but harder to manage

<!-- screenshot: Universe selector with BTC and SPX selected -->

#### 4. Configure Settings (JSON Editor)

**Profile JSON editor:** Shows default configuration

**Default configuration:**
```json
{
  "universe": ["BTC", "SPX"],
  "base_position_pct": 0.05,
  "max_position_pct": 0.10,
  "max_daily_loss_pct": 0.05,
  "slippage_pct": 0.01
}
```

**What each field means:**

**`base_position_pct`** (Default: 0.05 = 5%)
- **What it is:** Default trade size as percentage of portfolio
- **Example:** With $100,000 capital, 5% = $5,000 per trade
- **Range:** 0.01 (1%) to 0.20 (20%)
- **Conservative:** 0.02 (2%)
- **Balanced:** 0.05 (5%)
- **Aggressive:** 0.10 (10%)

**`max_position_pct`** (Default: 0.10 = 10%)
- **What it is:** Maximum capital allocated to single asset
- **Example:** With $100,000, 10% max = $10,000 limit per symbol
- **Purpose:** Prevents over-concentration in one asset
- **Should be:** 1.5-2x base_position_pct (allows position averaging)

**`max_daily_loss_pct`** (Default: 0.05 = 5%)
- **What it is:** Circuit breaker - stops trading if daily loss exceeds this
- **Example:** Lose $5,000 on $100k capital → trading pauses until next day
- **Purpose:** Prevents catastrophic drawdowns
- **Conservative:** 0.02 (2%)
- **Balanced:** 0.05 (5%)
- **Aggressive:** 0.10 (10%)

**`slippage_pct`** (Default: 0.01 = 1%)
- **What it is:** Simulated execution cost (buy higher, sell lower than signal price)
- **Example:** Signal says buy at $50,000 → actually buy at $50,500 (+1%)
- **Purpose:** Realistic backtest results (you never get perfect fills)
- **Typical values:** 0.005 (0.5%) for liquid markets, 0.02 (2%) for illiquid

<!-- screenshot: JSON editor with commented explanations -->

#### 5. Save the Profile

Click **"Save profile"** button

**Success message:** `Saved combined profile 'your-profile-name' to database.`

**Where it's stored:** PostgreSQL database, persists across sessions

**Versioning:** Each save creates a new version (allows rollback)

<!-- screenshot: Success message and profile appearing in saved profiles table -->

---

## Example Configurations

### Conservative Crypto Strategy

**Profile name:** `conservative-crypto`

```json
{
  "universe": ["BTC"],
  "base_position_pct": 0.02,
  "max_position_pct": 0.05,
  "max_daily_loss_pct": 0.02,
  "slippage_pct": 0.01
}
```

**Characteristics:**
- Single asset (BTC only)
- Small positions (2% of capital)
- Tight risk control (2% daily loss limit)
- Suitable for: Risk-averse traders, learning the system

### Balanced Multi-Asset Strategy

**Profile name:** `balanced-multi-asset`

```json
{
  "universe": ["BTC", "SPX", "CL"],
  "base_position_pct": 0.05,
  "max_position_pct": 0.10,
  "max_daily_loss_pct": 0.05,
  "slippage_pct": 0.01
}
```

**Characteristics:**
- Diversified across crypto, equities, commodities
- Standard position sizing (5%)
- Moderate risk limits
- Suitable for: Most users, general strategy testing

### Aggressive Cryptocurrency Strategy

**Profile name:** `aggressive-crypto`

```json
{
  "universe": ["BTC", "ETH"],
  "base_position_pct": 0.10,
  "max_position_pct": 0.20,
  "max_daily_loss_pct": 0.10,
  "slippage_pct": 0.015
}
```

**Characteristics:**
- Crypto-focused
- Large positions (10% base, 20% max)
- Relaxed risk limits (10% daily loss)
- Higher slippage assumption (1.5%)
- Suitable for: High risk tolerance, crypto specialists

---

## Model Presets

Model presets control which AI analyzes the markets.

**Location:** Configuration tab → Model Presets section

### Creating a Model Preset

**Fields:**

**1. Provider**
- `openai` - GPT models (most tested)
- `anthropic` - Claude models (good reasoning)
- `qwen` - Alibaba models (Chinese provider)

**2. Model Name**
- OpenAI: `gpt-4o-mini` (cheap, fast), `gpt-4o` (powerful, expensive)
- Anthropic: `claude-haiku-4-5-20251001` (fast), `claude-sonnet-4-5` (balanced)
- Qwen: `qwen3-max` (general), `qwen3-vl-plus` (vision)

**3. Temperature** (0.0 - 1.0)
- **0.0-0.2:** Deterministic, consistent decisions
- **0.3-0.5:** Balanced randomness
- **0.6-1.0:** Creative, varied outputs

**Recommended:** 0.1 for trading (consistency over creativity)

**4. Preset Name**
- Example: `fast-openai`, `cheap-haiku`, `premium-sonnet`

<!-- screenshot: Model preset form filled out -->

### Cost Considerations

**Cheap options (Development):**
- `gpt-4o-mini` (~$0.15 per 1M tokens)
- `claude-haiku` (~$0.25 per 1M tokens)

**Expensive options (Production):**
- `gpt-4o` (~$5.00 per 1M tokens)
- `claude-sonnet` (~$3.00 per 1M tokens)

**Rule of thumb:** Use cheap models for development and backtesting. Reserve expensive models for final validation or live paper trading.

**Backtest cost estimate:**
- 90-day backtest, 4-hour timeframe, 2 assets
- ~200 analysis calls
- With gpt-4o-mini: $0.50
- With gpt-4o: $15.00

---

## Managing Profiles

### Viewing Saved Profiles

**Location:** Configuration tab → Profiles table (bottom of page)

**Table columns:**
- **Source:** `db` (saved) or `session` (temporary)
- **Kind:** portfolio, risk, or combined
- **Name:** Profile identifier
- **Version:** Increments with each save

<!-- screenshot: Profiles table showing multiple saved profiles -->

### Loading a Profile

**To use in backtest:**
1. Go to Backtesting tab
2. Select profile from dropdown
3. Run backtest

**Profile settings are applied automatically**

### Editing a Profile

**To modify:**
1. Load profile in Configuration tab (type same name)
2. Edit JSON
3. Save (creates new version)

**Old version is preserved** for reproducibility

### Deleting a Profile

**Current limitation:** No delete button in MVP

**Workaround:**
- Profiles take minimal space
- Create new profile with different name
- Or manually delete from database

---

## Advanced Configuration

### Override Settings Per Symbol

**Not yet implemented in MVP**

**Future feature:** Different risk limits per asset
```json
{
  "universe": ["BTC", "SPX"],
  "base_position_pct": 0.05,
  "overrides": {
    "BTC": {"base_position_pct": 0.03},
    "SPX": {"base_position_pct": 0.07}
  }
}
```

### Dynamic Position Sizing

**Not yet implemented in MVP**

**Future feature:** Adjust size based on:
- Volatility (smaller positions in volatile markets)
- Confidence score (larger positions on high-confidence signals)
- Win streak (reduce after losses, increase after wins)

### Sector Exposure Limits

**Not yet implemented in MVP**

**Future feature:** Limit total exposure to categories
```json
{
  "sector_limits": {
    "crypto": 0.30,
    "equities": 0.50,
    "commodities": 0.20
  }
}
```

---

## Configuration Best Practices

### Start Conservative

**First profile should be low-risk:**
- Small position sizes (2-3%)
- Tight daily loss limit (2%)
- Single asset (BTC or SPX)

**Why:** Learn the system without risking too much capital (even in backtest)

### Test Multiple Variations

**Create 3 profiles for same strategy:**
1. `strategy-conservative` - 2% positions
2. `strategy-balanced` - 5% positions
3. `strategy-aggressive` - 10% positions

**Use Replay feature** to compare all three without re-running AI analysis

**See:** [Replay Documentation](dashboard.md#tab-5-replay)

### Match Configuration to Market Conditions

**High volatility (crypto):**
- Lower position sizes
- Wider slippage assumptions
- Tighter stop losses

**Low volatility (blue-chip stocks):**
- Standard position sizes
- Narrow slippage
- Moderate stop losses

### Keep Profiles Simple

**Avoid:**
- Complex nested overrides
- Too many assets (start with 1-3)
- Overly tight risk limits (blocks all trades)

**Good profile = Clear intent + Reasonable constraints**

---

## Troubleshooting

### "Saved profile but can't find it in backtest"

**Cause:** Profile saved as different type (portfolio vs combined)

**Solution:** Backtest dropdown shows only `combined` profiles. Check profile kind.

### "Backtest executed zero trades"

**Cause:** Risk limits too tight or position sizes too small

**Check:**
- `max_position_pct` > `base_position_pct`
- `max_daily_loss_pct` not too low (try 10% for testing)
- Assets in universe actually have data for date range

### JSON validation error

**Cause:** Syntax error in JSON (missing comma, quote, bracket)

**Solution:**
- Copy JSON to https://jsonlint.com for validation
- Check all keys are quoted: `"base_position_pct"` not `base_position_pct`
- Ensure commas between fields (but not after last field)

### Profile disappeared after refresh

**Cause:** Saved to session instead of database

**Solution:**
- Verify database connection (check Dashboard for green indicator)
- Re-save profile
- Check **Profiles** table shows `source: db`

---

## Real-World Configuration Examples

### Example 1: Learning the System

**Scenario:** Brand new user, wants to understand how it works

**Profile:**
```json
{
  "universe": ["BTC"],
  "base_position_pct": 0.01,
  "max_position_pct": 0.02,
  "max_daily_loss_pct": 0.05,
  "slippage_pct": 0.01
}
```

**Rationale:**
- Single asset (less to track)
- Tiny positions (1% = learning, not serious testing)
- Run 7-day backtest for quick feedback

### Example 2: Validating a Strategy Idea

**Scenario:** Have a hypothesis about 4-hour BTC trends

**Profile:**
```json
{
  "universe": ["BTC"],
  "base_position_pct": 0.05,
  "max_position_pct": 0.10,
  "max_daily_loss_pct": 0.05,
  "slippage_pct": 0.01
}
```

**Testing approach:**
1. 30-day backtest (quick check)
2. If promising, 90-day backtest
3. If still good, 180-day backtest
4. If passing viability criteria, paper trade

### Example 3: Building a Diversified Portfolio

**Scenario:** Want uncorrelated assets for steady returns

**Profile:**
```json
{
  "universe": ["BTC", "SPX", "CL"],
  "base_position_pct": 0.04,
  "max_position_pct": 0.08,
  "max_daily_loss_pct": 0.06,
  "slippage_pct": 0.01
}
```

**Rationale:**
- Three uncorrelated assets (crypto, equities, commodities)
- Slightly smaller positions (4%) to allow multiple open positions
- Moderate risk (6% daily loss limit)

---

## Related Documentation

- **Configuration Architecture**: [CONFIGURATION.md](../03_design/CONFIGURATION.md) - QuantAgent-cxu
- **Risk Management**: [Trading System Requirements](../01_requirements/trading_system_requirements.md)
- **Position Monitoring**: [Active Position Monitoring](../03_design/QuantAgent-nu7-DS-active-position-monitoring.md) - QuantAgent-nu7
- **Strategy Abstraction**: [QuantAgent-enn Implementation](../06_implementation/QuantAgent-enn-IM-trading-strategy.md) - QuantAgent-enn
- **Backtesting Integration**: [QuantAgent-on4 Planning](../02_planning/phase1_roadmap.md) - QuantAgent-on4

---

*Configuration is iterative. Start simple, test thoroughly, refine based on results.*
