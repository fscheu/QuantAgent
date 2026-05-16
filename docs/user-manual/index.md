# QuantAgent User Manual

**A multi-agent AI system for automated trading strategy development and validation**

Last updated: May 16, 2026

---

## What is QuantAgent?

QuantAgent uses four specialized AI agents to analyze financial markets and make trading decisions. The system validates strategies through backtesting on historical data and can execute simulated trades (paper trading) without risking real money.

**Key capabilities:**
- Automated market analysis using AI agents
- Strategy backtesting on historical data
- Paper trading simulation
- Risk management and position monitoring
- Performance tracking and metrics

---

## Table of Contents

### Getting Started
- **[Getting Started Guide](getting-started.md)** - Install and run your first backtest

### Core Features
- **[Dashboard](dashboard.md)** - Navigate the web interface
- **[Backtesting](backtesting.md)** - Test strategies on historical data
- **[Strategy Configuration](strategy-configuration.md)** - Set up trading rules and risk limits
- **[Analysis & Signals](analysis-and-signals.md)** - Understand AI agent decisions
- **[Monitoring](monitoring.md)** - Track performance and system logs
- **[Paper Trading](monitoring.md#paper-trading-tab)** - Monitor scheduler health, positions, orders, P&L, and LLM telemetry
- **[Paper Trading Automation](paper-trading-automation.md)** - Keep the scheduler running 24/7
- **[Profile CLI](profile-cli.md)** - Manage strategy profiles from the terminal

---

## Who Should Use This Manual?

This guide is for **strategy developers** and **traders** who want to:
- Test trading ideas without coding
- Validate strategies before risking capital
- Understand AI-driven market analysis
- Monitor automated trading systems

**No programming required** for basic usage. Advanced features may need Python knowledge.

---

## Quick Links

- [Installation Prerequisites](getting-started.md#prerequisites)
- [Run Your First Backtest](backtesting.md#running-a-backtest)
- [Create a Strategy Profile](strategy-configuration.md#creating-profiles)
- [View Analysis Results](analysis-and-signals.md#viewing-signals)
- [Check Performance Metrics](backtesting.md#understanding-metrics)
- [Set Up Paper Trading Automation](paper-trading-automation.md)
- [Automate Profile Management](profile-cli.md)

---

## Need Help?

- **Technical Documentation**: See [docs/03_design/](../03_design/) for architecture details
- **Issue Tracking**: Check [.beads/](.beads/) for known issues and roadmap
- **Source Code**: [GitHub Repository](https://github.com/fscheu/QuantAgent)

---

*Built with LangGraph and powered by OpenAI/Anthropic models*
