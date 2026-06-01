# mibel-trading

Trading sobre **MIBEL DAM + servicios de ajuste** combinando estrategia rule-based y agente PPO. Quinta pieza del portfolio MIBEL, adaptación de la tesis doctoral de **Demir (2023, TU/e)** sobre _Statistical Arbitrage Trading on Electricity Markets Using Deep Reinforcement Learning_, originalmente desarrollada sobre los mercados CID + BAL holandeses.

## Contexto

Demir entrenó agentes A3C y A2C sobre Continuous Intraday + Balancing del mercado holandés con datos privados de Scholt Energy (limit order book + trade book), obteniendo PnL de €19.927 (A3C) y €97.853 (A2C+RB) sobre 7.017–33.805 MWh en contratos 2020.

Este repo adapta la metodología a **datos públicos del MIBEL**:

- **DAM**: precio horario OMIE (España).
- **Servicios de ajuste**: regulación secundaria (banda + energía, ESIOS 634/682/683), terciaria (ESIOS 676/677), desvíos (ESIOS 686/687).

No se replica el módulo de Continuous Intraday porque OMIE no publica el limit order book del MIC.

## Conexión con el portfolio

Quinta pieza del programa de investigación sobre MIBEL:

1. [mibel-forecasting](https://github.com/cmvdata/mibel-forecasting): LEAR DAM forecasting.
2. [mibel-derivatives](https://github.com/cmvdata/mibel-derivatives): jump-diffusion + Schwartz-Smith, swing + tolling + PPA.
3. [mibel-risk](https://github.com/cmvdata/mibel-risk): VaR portfolio + PFE + CVA.
4. **mibel-trading** (este repo): rule-based + PPO sobre DAM + servicios de ajuste.

Referencia regulatoria adicional: paper Vilches (2026) sobre microestructura MIBEL y caso CNMC SNC/DE/017/23.

## Estructura

src/mibel_trading/
├── env/          # Gymnasium environment
├── data/         # OMIE + ESIOS ingestion
├── strategies/   # Rule-based DAM→ajuste
├── benchmarks/   # BENCHVWAP, BENCH, BENCHPLUS
├── agents/       # PPO with stable-baselines3
└── eval/         # Walk-forward CV, PnL/PD/PT

## Roadmap

| Pieza | Estado |
|---|---|
| 0. Scaffold + CI | ✅ |
| 1. Gymnasium env + data loaders | 🔄 |
| 2. Rule-based strategy DAM→ajuste | ⏳ |
| 3. Benchmarks BENCHVWAP/BENCH/BENCHPLUS | ⏳ |
| 4. PPO agent + behaviour cloning | ⏳ |
| 5. Walk-forward evaluation + report | ⏳ |

## Limitaciones declaradas

- Sin acceso a limit order book del MIC español (datos no públicos en OMIE).
- Datos agregados horarios, no tick-level.
- Costes de transacción simplificados (tarifa OMIE estándar).

## Referencias

- Demir, S. (2023). _Statistical Arbitrage Trading on Electricity Markets Using Deep Reinforcement Learning_. PhD Thesis, Eindhoven University of Technology.
- Lago, J., Marcjasz, G., De Schutter, B., & Weron, R. (2021). _Forecasting day-ahead electricity prices_. Applied Energy, 293.
- Vilches, C. (2026). _Detecting Structural Manipulation in Electricity Markets_. Universitat de Barcelona.
- CNMC (2024). Resolución SNC/DE/017/23.

## Reproducibilidad

uv sync --extra dev
uv run pytest

## Licencia

MIT
