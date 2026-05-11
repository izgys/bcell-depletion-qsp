# BCell-Depletion-QSP — Mechanistic modelling of B cell depletion and recovery under anti-CD20 therapy

**Author:** Iker Zapirain Gysling  
**Status:** Active development  
**Domain:** Translational Pharmacology · Quantitative Systems Pharmacology · Drug Discovery

---

## The Problem

B cell-depleting therapies — most commonly anti-CD20 monoclonal antibodies such as rituximab and ocrelizumab — are cornerstones of treatment in rheumatoid arthritis, multiple sclerosis, and B cell malignancies. Yet key translational questions remain difficult to address experimentally:

- How deep is tissue depletion relative to peripheral blood, and how does this vary across dosing regimens?
- What biological parameters most strongly govern the timing and extent of B cell repopulation after treatment ends?
- How do dose magnitude and dosing interval interact to sustain target attainment over time?

Mechanistic, ODE-based modelling offers a tractable route to these questions — enabling controlled in-silico experiments that are infeasible or unethical in patients.

---

## What It Does

This repository implements a multi-compartment quantitative systems pharmacology (QSP) model of B cell dynamics under anti-CD20 therapy. The model captures:

- B cell maturation and export from bone marrow
- Peripheral blood trafficking and natural turnover
- Lymphoid tissue (spleen/lymph node) B cell pools and recirculation
- Drug pharmacokinetics (two-compartment IV model)
- Drug pharmacodynamics (sigmoidal Emax killing function)
- B cell repopulation following treatment cessation

Parameters are estimated from literature data using Bayesian inference (PyMC, NUTS sampler), with explicit uncertainty quantification and posterior predictive validation.

---

## Architecture

```
Bone Marrow
    │
    │  maturation + export
    ▼
Peripheral Blood ◄──────────────── Drug PK (two-compartment)
    │        ▲                          │
    │        │  recirculation           │ Emax PD (killing)
    ▼        │                          ▼
Lymphoid Tissue                   B cell depletion
(Spleen / LN)                     (blood + tissue)
```

**Key components:**

| Module | Description |
|---|---|
| `src/model/odes.py` | ODE system (5 state variables) |
| `src/model/pk.py` | 1- and 2-compartment PK models |
| `src/model/pd.py` | Emax, linear, and sigmoidal PD models |
| `src/model/dosing.py` | Discontinuous dosing event handler |
| `src/simulation/simulator.py` | Forward simulation engine (scipy LSODA) |
| `src/estimation/bayesian.py` | PyMC-based Bayesian parameter estimation |
| `src/simulation/sensitivity.py` | Local and global (Sobol) sensitivity analysis |

---

## State Variables and ODE System

### State Variables

| Variable | Description | Units |
|---|---|---|
| `B_bm` | Bone marrow B cell precursor pool | cells/µL |
| `B_blood` | Peripheral blood B cells | cells/µL |
| `B_tissue` | Lymphoid tissue B cells | cells/µL |
| `C_central` | Drug concentration, central compartment | ng/mL |
| `C_peripheral` | Drug concentration, peripheral compartment | ng/mL |

### ODE System

**Bone marrow:**
```
dB_bm/dt = k_prod - k_export · B_bm - d_bm · B_bm
```

**Peripheral blood:**
```
dB_blood/dt = k_export · B_bm
              - k_tissue · B_blood
              + k_recirculate · B_tissue
              - d_blood · B_blood
              - kill(C) · B_blood
```

**Lymphoid tissue:**
```
dB_tissue/dt = k_tissue · B_blood
               - k_recirculate · B_tissue
               - d_tissue · B_tissue
               - kill(C) · B_tissue · f_tissue
```

**PK (two-compartment IV):**
```
dC_central/dt    = Dose(t)/V1 - (CL/V1 + Q/V1) · C_central + (Q/V2) · C_peripheral
dC_peripheral/dt = (Q/V1) · C_central - (Q/V2) · C_peripheral
```

**PD (sigmoidal Emax):**
```
kill(C) = Emax · C^n / (EC50^n + C^n)
```

Steady-state initial conditions are computed analytically from the parameter set before drug introduction.

### Key Parameters

| Parameter | Description | Prior / Value | Literature basis |
|---|---|---|---|
| `k_export` | BM→blood export rate | LogNormal(μ=0.1, σ=0.3)/day | Bleyzac et al., 2001 |
| `d_blood` | Blood B cell turnover | LogNormal(μ=0.03, σ=0.3)/day | Tough & Sprent, 1994 |
| `d_tissue` | Tissue B cell turnover | LogNormal(μ=0.01, σ=0.3)/day | Macallan et al., 2005 |
| `k_tissue` | Blood→tissue trafficking rate | LogNormal(μ=0.5, σ=0.4)/day | Shen et al., 2004 |
| `k_recirculate` | Tissue→blood recirculation rate | LogNormal(μ=0.3, σ=0.4)/day | Shen et al., 2004 |
| `Emax` | Maximum killing rate | Beta(α=8, β=2) | Fitted |
| `EC50` | Drug conc. for 50% max killing | LogNormal(μ=10, σ=0.5) ng/mL | Quartier et al., 2003 |
| `n` | Hill coefficient | LogNormal(μ=1, σ=0.3) | Fitted |
| `CL` | Drug clearance | Drug-specific | Mager & Jusko, 2001 |
| `V1`, `V2` | Distribution volumes | Drug-specific | Mager & Jusko, 2001 |
| `Q` | Inter-compartmental flow | Drug-specific | Mager & Jusko, 2001 |

---

## Scientific Background

### B Cell Biology

B cells are central mediators of humoral immunity. They originate from haematopoietic stem cells in the bone marrow, mature through pro-B, pre-B, immature, and transitional stages before entering the peripheral circulation as naïve B cells. From there they home to secondary lymphoid organs — spleen, lymph nodes, and mucosa-associated lymphoid tissue — where antigen encounter drives differentiation into short-lived plasmablasts or long-lived plasma cells, with a subset forming memory B cells.

Dysregulation of B cell homeostasis underlies a broad range of immune-mediated diseases, including rheumatoid arthritis, systemic lupus erythematosus, multiple sclerosis, and certain B cell malignancies. Therapeutic B cell depletion — most commonly via anti-CD20 monoclonal antibodies such as rituximab and ocrelizumab — is a cornerstone of treatment in several of these conditions.

### Modelling Rationale

Multi-compartment ODE models offer a principled framework for addressing translational questions that are difficult to answer experimentally: specifically, how peripheral blood measurements relate to tissue depletion depth, and which biological parameters most strongly determine repopulation kinetics. This model formalises those relationships in a physiologically grounded structure, enabling controlled in-silico dosing experiments and parameter sensitivity analysis.

---

## Methodology

### Forward Simulation

The ODE system is integrated using `scipy.integrate.solve_ivp` with the `LSODA` solver, which handles the stiffness arising from fast drug kinetics alongside slow B cell turnover. Dosing events are implemented as discontinuous forcing functions. Steady-state initial conditions are derived analytically before drug introduction.

### Bayesian Parameter Estimation

Parameters are estimated using PyMC with the NUTS (No-U-Turn) sampler. Physiologically informed priors are placed on all fitted parameters based on published B cell kinetic studies. Posterior distributions are sampled with 2000 draws (1000 tuning steps) at target acceptance rate 0.9.

```python
with pm.Model() as qsp_model:
    # Physiologically informed priors
    d_blood = pm.LogNormal("d_blood", mu=np.log(0.03), sigma=0.3)
    EC50    = pm.LogNormal("EC50",    mu=np.log(10.0),  sigma=0.5)
    Emax    = pm.Beta("Emax", alpha=8, beta=2)

    # Likelihood
    B_pred = simulate_model(params=[d_blood, EC50, Emax, ...])
    sigma  = pm.HalfNormal("sigma", sigma=0.1)
    obs    = pm.Normal("obs", mu=B_pred, sigma=sigma, observed=B_obs)

    trace = pm.sample(2000, tune=1000, target_accept=0.9)
```

MCMC diagnostics (R-hat < 1.01, ESS > 400) are reported for all parameters. Posterior predictive checks validate model adequacy before any downstream analysis.

### Uncertainty Quantification

Two sources of uncertainty are explicitly propagated:

- **Parameter uncertainty** — credible intervals (50% and 95%) derived from the posterior distribution
- **Structural uncertainty** — comparison of alternative model structures (one- vs two-compartment PK; linear vs sigmoidal PD)

All model predictions are reported with credible intervals, not point estimates.

### Sensitivity Analysis *(v2)*

**Local sensitivity analysis** — partial derivatives of model outputs with respect to each parameter at the posterior median. Identifies which parameters govern early depletion vs. recovery kinetics.

**Global sensitivity analysis** — Sobol indices via SALib, capturing nonlinear and interaction effects across the full parameter space.

---

## Evaluation

### Validation Strategy

The model is validated at three levels before any predictive use:

**1. Synthetic data recovery**
Parameters are estimated from synthetic datasets generated at known ground-truth values. This confirms parameter identifiability and correctness of the estimation procedure before any real data is used. Reported metrics: posterior median bias, 95% credible interval coverage, R-hat, ESS.

**2. Literature data fitting**
The model is fitted to digitised B cell count data from published clinical studies (rituximab in RA, ocrelizumab in MS). Reported metrics: posterior predictive coverage, WAIC for model comparison across structural alternatives.

**3. Cross-validation**
A model fitted to one dataset is used to predict an independent dataset. Assesses generalisation beyond the training data.

### Known Failure Modes

- **Parameter non-identifiability**: trafficking rates `k_tissue` and `k_recirculate` may be weakly identified from blood-only data. Tissue compartment data would substantially constrain these.
- **Sparse data limitation**: typical clinical B cell count datasets have 5–10 timepoints. Posterior distributions on fast-timescale parameters (PK) are well-constrained; slow-timescale parameters (repopulation) may remain wide.
- **Structural assumptions**: the model assumes a single well-mixed lymphoid tissue compartment. Spleen, lymph node, and MALT likely differ in kinetics — compartmental lumping introduces structural error.
- **Drug-specificity**: PK parameters (CL, V1, V2, Q) are drug-specific and taken from the literature. Variability in PK across patients is not currently modelled (population PK extension is a v2 target).

---

## Key Design Decisions

**Why ODE-based rather than agent-based?**
Agent-based models can capture cell-level stochasticity but are computationally expensive and harder to fit to sparse clinical data. For the translational questions addressed here — compartment-level depletion kinetics and dosing regimen comparison — deterministic ODEs are sufficient, interpretable, and identifiable.

**Why PyMC / NUTS rather than maximum likelihood?**
Maximum likelihood estimation gives point estimates; NUTS gives full posterior distributions. For a model with 8+ correlated parameters fitted to sparse data, posterior distributions are essential to characterise uncertainty — both in parameters and in downstream predictions. Credible intervals on predicted depletion trajectories are a primary output of the model.

**Why LSODA rather than a fixed-step solver?**
The ODE system is stiff: drug concentrations change on a timescale of hours, while B cell repopulation occurs over weeks to months. LSODA's adaptive step-size control handles this automatically without manual tuning of integration parameters.

**Why config-driven parameters?**
All model parameters, dosing schedules, and estimation settings are defined in YAML files under `config/`. This enables reproducible runs, easy drug-switching (rituximab → ocrelizumab requires only a different PK config), and MLflow parameter logging without code changes.

---

## Quickstart

### With Docker (recommended)

```bash
git clone https://github.com/izgys/bcell-depletion-qsp
cd bcell-depletion-qsp
docker build -t bcell-depletion-qsp .
docker run -p 8888:8888 bcell-depletion-qsp
```

### Local installation

```bash
git clone https://github.com/izgys/bcell-depletion-qsp
cd bcell-depletion-qsp
conda env create -f environment.yml
conda activate bcell-depletion-qsp
pip install -e .
```

### Running a simulation

```bash
python -m src.simulation.simulator \
    --config config/model_params.yaml \
    --pk config/pk_rituximab.yaml \
    --dosing config/dosing_schedules.yaml \
    --output results/rituximab_sim
```

### Running parameter estimation

```bash
python -m src.estimation.bayesian \
    --config config/estimation_config.yaml \
    --data data/processed/rituximab_blood_depletion.csv \
    --output results/estimation
```

### MLflow tracking

All runs are logged to MLflow automatically — parameters, diagnostics, and output figures.

```bash
mlflow ui
# open http://localhost:5000
```

---

## Repository Structure

```
bcell-depletion-qsp/
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
├── Dockerfile
│
├── config/
│   ├── model_params.yaml          # Physiological parameters and priors
│   ├── pk_rituximab.yaml          # Rituximab PK (CL, V1, V2, Q)
│   ├── pk_ocrelizumab.yaml        # Ocrelizumab PK
│   ├── dosing_schedules.yaml      # Standard dosing regimens
│   └── estimation_config.yaml     # MCMC settings (draws, tune, target_accept)
│
├── src/
│   ├── model/
│   │   ├── odes.py                # ODE system
│   │   ├── pk.py                  # PK model
│   │   ├── pd.py                  # PD model (Emax, linear, sigmoidal)
│   │   └── dosing.py              # Dosing event handler
│   ├── estimation/
│   │   ├── bayesian.py            # PyMC estimation
│   │   ├── priors.py              # Prior distributions
│   │   └── diagnostics.py        # R-hat, ESS, trace plots
│   ├── simulation/
│   │   ├── simulator.py           # Forward simulation engine
│   │   ├── sensitivity.py         # Sensitivity analysis (v2)
│   │   └── dose_optimisation.py   # Dose optimisation (v2)
│   └── visualisation/
│       ├── trajectories.py        # B cell and drug concentration plots
│       └── uncertainty.py         # Credible interval plots
│
├── data/
│   ├── raw/                       # Digitised literature data (with source DOIs)
│   └── processed/                 # Cleaned, analysis-ready datasets
│
├── notebooks/
│   ├── 01_model_exploration.ipynb
│   ├── 02_parameter_estimation.ipynb
│   ├── 03_uncertainty_quantification.ipynb
│   ├── 04_sensitivity_analysis.ipynb
│   └── 05_dose_optimisation.ipynb
│
├── experiments/
│   ├── synthetic_validation/      # Parameter recovery from synthetic data
│   ├── rituximab_RA/              # Rituximab in rheumatoid arthritis
│   └── ocrelizumab_MS/            # Ocrelizumab in multiple sclerosis (v2)
│
├── tests/
│   ├── test_odes.py
│   ├── test_pk.py
│   └── test_simulator.py
│
└── .github/
    └── workflows/
        └── ci.yml                 # Linting + smoke tests
```

---

## Roadmap

**v1 (current)**
- [x] ODE model structure
- [x] Forward simulation engine with LSODA
- [x] Config-driven parameter management (YAML + Hydra)
- [x] MLflow experiment tracking
- [ ] Bayesian parameter estimation (PyMC, NUTS) — in progress
- [ ] Posterior predictive checks and MCMC diagnostics
- [ ] Synthetic data recovery case study
- [ ] Rituximab RA case study (literature data fitting)
- [ ] Docker packaging and CI

**v2 (planned)**
- [ ] Global sensitivity analysis (Sobol indices, SALib)
- [ ] Dose optimisation module
- [ ] Ocrelizumab MS case study
- [ ] Population PK extension (inter-individual variability)
- [ ] Streamlit dashboard for interactive exploration

---

## Scientific References

- Bleyzac N et al. (2001). *Clin Pharmacokinet.* — B cell turnover kinetics in humans. DOI: 10.2165/00003088-200140020-00004
- Tough DF & Sprent J (1994). *J Exp Med.* — Lymphocyte lifespan and memory. DOI: 10.1084/jem.179.4.1127
- Macallan DC et al. (2005). *J Exp Med.* — B lymphocyte kinetics in humans. DOI: 10.1084/jem.20050366
- Quartier P et al. (2003). *Arthritis Rheum.* — Rituximab PK/PD in paediatric populations. DOI: 10.1002/art.11100
- Genovese MC et al. (2008). *Arthritis Rheum.* — B cell repopulation kinetics post-rituximab. DOI: 10.1002/art.23439
- Shen et al. (2004). *J Immunol.* — Multi-compartment B cell trafficking model. DOI: 10.4049/jimmunol.172.3.1454
- Mager DE & Jusko WJ (2001). *Pharm Res.* — Target-mediated drug disposition framework. DOI: 10.1023/A:1013349870148

---

## Author

**Iker Zapirain Gysling**  
Computational Biochemist, PhD  
Barcelona, Spain  
[LinkedIn](https://linkedin.com/in/zgysling) · [GitHub](https://github.com/izgys)

---

## License

MIT License — see `LICENSE` for details.
