<picture>
  <source media="(prefers-color-scheme: dark)" srcset=".github/assets/banner-dark.png">
  <img alt="The University of Chicago, MS in Applied Data Science. ADSP 32014 course project." src=".github/assets/banner-light.png">
</picture>

# Bayesian Modeling of Sepsis Progression in ICU Patients Using MIMIC-IV

> **Example project from Winter 2026.** This is a copy of [regularize-this/sepsis-progression-modeling](https://github.com/regularize-this/sepsis-progression-modeling), shown with the authors' work and commit history intact. Their own write-up is [REPORT.md](REPORT.md), with the cohort rationale in [docs/dataset.md](docs/dataset.md) and their takeaways in [docs/discussion.md](docs/discussion.md). Course staff wrote this README and made the edits listed under [Changes by course staff](#changes-by-course-staff), including the section on [what to learn from this project](#what-to-learn-from-this-project). The project uses MIMIC-IV, which requires credentialed access. No data is in this repository, and course staff redacted three patient identifiers from saved outputs.

Sepsis patients in an ICU move through stages that nobody observes directly, from infection through organ dysfunction to shock or recovery, and every hour of delayed treatment matters. Can a hidden Markov model infer those stages from the vitals, labs and treatments charted in the ICU, and say how a treatment changes the odds of getting worse? The team built two. The first is a Gaussian HMM on hourly vitals, fit by expectation maximization as a baseline. The second is a Bayesian HMM in PyMC on 1,502 MIMIC-IV stays whose transition probabilities depend on the treatments in progress, the patient's severity on arrival and the time since the last observation, with a decision-support layer that compares treatment scenarios for a patient and asks how long treatment can safely wait.

## Summary

The cohort is every adult aged 18 to 65 in MIMIC-IV whose first ICU stay lasted between 12 hours and 10 days and who received antibiotics and had cultures taken within an hour of each other, the team's proxy for suspected sepsis: 1,502 stays, 17 percent of which ended in death in hospital. Two pipelines extract it, one in DuckDB for Model A and one in chunked pandas for Model B, and agree to within one patient.

**Model A** resamples five vitals and labs to an hourly grid, fits a four-state Gaussian HMM with hmmlearn, names the states by sorting on mean lactate, flags any hour in which the chance of moving to a worse state exceeds 30 percent, and attaches treatment suggestions by comparing vitals in treated and untreated hours of the same state. Only two figures of it survive in the repository; its model file was never committed and the training script does not run.

**Model B** keeps every charted observation at its own timestamp, 284,673 of them, with a median gap of 35 minutes, computes a SOFA organ-failure score at each one, and models 14 standardized features with three hidden states. The transition logits between states are a baseline plus terms for antibiotics, vasopressors and IV fluids, each treatment's interaction with baseline SOFA, baseline SOFA itself, and the time gap. The forward algorithm is written by hand and vectorized over all patients, and the posterior is approximated by ADVI in six hours. States sorted by SOFA come out as Mild, Moderate and Shock, with the Shock state marked by high lactate, creatinine and bilirubin and low platelets and blood pressure. A decision-support notebook infers each patient's state, compares seven treatment scenarios over 48 hours with intervals from posterior draws, sweeps treatment delay, and splits 500 patients into risk groups whose in-hospital mortality differs 6.5-fold.

The staff section below explains why the largest treatment effect is partly built into the state definition, why the counterfactuals step outside the data, why the validation is in-sample and near-circular, and why every state probability the model reports is 100 or 0 percent.

## Team and contributions

Team "Regularize This!", Winter 2026.

- **Hoon Yum** ([@jhoonyum](https://github.com/jhoonyum)): the six Model B notebooks, from extraction through the PyMC model, ADVI fit and decision-support layer, and the write-up's Model B section. He appears as Junghoon Yum in the authors' list.
- **Ryan Abdelrahim** ([@alraiana](https://github.com/alraiana)): the repository layout, the DuckDB cohort extraction and hourly feature pipeline for Model A, the dataset document with its clinical references, the discussion, and the write-up's clinical framing.
- **Collin Kim** ([@collin-k](https://github.com/collin-k)): the repository, license and ignore file; Model A's training script and evaluation module, including the alert and intervention logic; revisions to the discussion.
- **Yuta Kobayashi** ([@yutakoba120](https://github.com/yutakoba120)): Model A's architecture and inference diagrams and its example figures, and the write-up's Model A section.
- **Nick Thompson** ([@nsthompson-ux](https://github.com/nsthompson-ux)): the Markov-assumption exploratory analysis for Model A, and a notebook that re-implements Model A's missing pieces so that it can be run.

The repository ran from February 13 to March 24, 2026, with nearly all of the work landing between March 12 and 16. The attributions follow the [commit history](https://github.com/uchicago-dsi-bayesianml/wi26-sepsis-progression-hmm/commits/main).

## Getting started

Everything here depends on MIMIC-IV, which is not public. Access requires a [PhysioNet](https://physionet.org/content/mimiciv/) account, the CITI human-subjects course and a signed data use agreement, and the download is about 70 GB of CSV files. Without it, read the notebooks: Model B's six notebooks keep their outputs, so the whole pipeline can be followed on screen.

The authors' `requirements.txt` lists the core packages; the notebooks also use seaborn, statsmodels, hmmlearn and SciPy.

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt seaborn statsmodels hmmlearn scipy jupyter
```

**Model B.** Run `notebooks/01` through `06` in order. Each has a configuration cell at the top with the authors' local paths; point them at your MIMIC-IV folder and a working folder. Extraction streams the 42 GB `chartevents` file in chunks and takes a few minutes; preprocessing and feature engineering take under a minute; the ADVI fit took the authors 364 minutes for 50,000 iterations on a laptop, and the decision-support notebook reloads the saved trace.

**Model A.** As committed, it does not run. `src/train_model_1.py` imports a `model` module that was never added to the repository, and `src/data_processing.py` is a notebook export with a blank where the data path belongs, so Python cannot parse it. `notebooks/model_results.ipynb` re-implements the missing pieces against the hourly table that the processing script was meant to produce, but has no saved outputs.

## Data

Data files are never committed to this repository. Git ignores `data/` and every CSV.

- **Source:** [MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/) (Johnson et al., *Scientific Data*, 2023), de-identified records of Beth Israel Deaconess Medical Center from 2008 to 2022: about 94,000 ICU stays. Model B reads `icustays`, `patients`, `admissions`, `chartevents`, `labevents`, `inputevents`, `procedureevents`, `outputevents` and `prescriptions`.
- **Terms:** PhysioNet's credentialed health data license. The data may not be shared or redistributed, and no attempt may be made to identify anyone. Dates in MIMIC are shifted into the future per patient; the subject IDs are not patient identifiers in the ordinary sense, but individual-level rows are still the data, which is why course staff removed the three that had been saved in outputs.
- **Cohort:** first ICU stay per patient; stay between 12 hours and 10 days; age 18 to 65; an intravenous antibiotic from a list of 18 and a culture from a list of 8 within an hour of each other during the stay. From 94,458 stays to 65,366 first stays, 58,506 by duration, 29,421 by age, and 1,502 by the sepsis criterion. The item IDs and the clinical reasoning are in [docs/dataset.md](docs/dataset.md) and in the first notebook.
- **Model B's table:** one row per charted observation, 284,673 rows, a median of 170 per stay. Vitals are forward-filled for up to two hours and labs without limit; what is still missing is set to the column median, and a missing SOFA component counts as zero.
- **Model A's table:** one row per hour. Heart rate, MAP and SpO2 are filled forward and backward; WBC is carried for 12 hours and lactate for 6, after which the mean is used and a missing flag is set.

## Project structure

| Path | Contents |
|---|---|
| `REPORT.md` | The authors' write-up: the clinical problem, Model A's inputs and modules, Model B's design and outputs, installation, future work. |
| `docs/dataset.md`, `docs/discussion.md` | The cohort definition with references, and the team's takeaways and limitations. |
| `notebooks/01_Data_Extraction.ipynb` | Model B: the cohort and chunked extraction of vitals, labs, treatments and admissions from MIMIC-IV. |
| `notebooks/02_Data_Preprocessing.ipynb` | Model B: cleaning, unit conversion, the per-observation timeline, treatment flags, time gaps. |
| `notebooks/03_EDA.ipynb` | Model B: demographics, time gaps, distributions, missingness, treatment timing, survivors against non-survivors. |
| `notebooks/04_SOFA_Feature_Engineering.ipynb` | Model B: SOFA at every observation, baseline SOFA, scaling, sequence arrays. |
| `notebooks/05_Bayesian_HMM_ADVI.ipynb` | Model B: the PyMC model, the vectorized forward algorithm, the ADVI fit, state profiles, treatment effects, a 48-hour counterfactual. |
| `notebooks/06_Prediction_DSS.ipynb` | Model B: per-patient state inference, treatment comparison with intervals, delay sweeps, risk stratification. |
| `notebooks/exploratory_analysis.ipynb` | Model A: sequence lengths, an empirical transition matrix over oxygen and vasopressor states, dwell times, autocorrelation, stationarity tests. |
| `notebooks/model_results.ipynb` | Model A: a self-contained re-implementation of the training and evaluation, without saved outputs. |
| `src/data_processing.py` | Model A: the DuckDB cohort query and hourly feature construction, as a script export. |
| `src/model_1.py`, `src/train_model_1.py`, `src/evaluation.py` | Model A: data preparation, the training entry point, and the state naming, alert and intervention logic. |
| `figures/` | Model A's architecture and inference diagrams, one patient timeline, and the clinical-pathway illustration. |

## Method

**Model A.** Five features per hour, heart rate, SpO2, white cell count, lactate and MAP, clipped to physiological ranges and standardized on the training patients, with stale lactate and WBC values masked to the mean after two hours. A four-state Gaussian HMM with full covariances, fit by expectation maximization in hmmlearn on 80 percent of patients. States are named Normal, Sepsis, Severe Sepsis and Septic Shock in order of mean lactate. For each hour, the filtered state distribution is pushed through the transition matrix, and the mass on worse states is the progression probability; above 30 percent an alert fires. Treatment suggestions compare mean vitals in treated and untreated hours of the same state and recommend a treatment when the treated mean looks healthier.

**Model B, emissions.** Fourteen features per observation, standardized: heart rate, systolic and mean arterial pressure, temperature, respiratory rate, SpO2, lactate, white cells, platelets, creatinine, bilirubin, hemoglobin, the SOFA total computed at that observation, and the time since the previous observation. Each of three states has its own mean and standard deviation per feature, with independent Gaussian emissions.

**Model B, transitions.** For a patient in state *i*, the logit of moving to state *j* is

> logit[i, j] = base[i, j] + Σ_d treatment_d · (β_treat[d, i, j] + β_int[d, i, j] · baselineSOFA) + β_sev[i, j] · baselineSOFA + β_time[i, j] · Δt

with a row-wise softmax, where *d* runs over antibiotics, vasopressors and IV fluids as 0/1 flags and baseline SOFA is standardized. Priors: the initial state distribution is Dirichlet(2, 2, 2); emission means are Normal(0, 2) and scales HalfNormal(1); base logits are Normal(0, 1); treatment, severity and time coefficients are Normal(0, 0.5) and the interactions Normal(0, 0.3).

**Model B, inference.** The hidden states are marginalized by the forward algorithm, written in log space with a PyTensor scan over time and a mask for sequences shorter than the padded length, and the summed log-likelihood enters the model as a potential. Sequences longer than 200 observations are truncated. ADVI runs for 50,000 iterations with Adam at a learning rate of 0.01, and 2,000 draws are taken from the fitted approximation. The fit took 364 minutes.

**Decision support.** For one patient, the forward algorithm over the full history gives the current state probabilities, once per posterior draw. From that state, 48 hourly transition steps under each of seven treatment scenarios give a 48-hour distribution over states, summarized as shock risk with a 95 percent interval across draws; the lowest-risk scenario that does not involve vasopressors is recommended. A delay sweep simulates *d* hours untreated followed by the best treatment and looks for the delay at which the curve bends. Over 500 patients, a current shock probability above 30 percent defines a high-risk group.

### What is Bayesian here

Model B is a Bayesian HMM in the full sense. Every parameter, the initial distribution, the emission means and scales, the baseline transition logits and the treatment, severity and time coefficients, has a prior, and the hidden state sequence is integrated out exactly by the forward algorithm, which is the evaluation problem of the course's HMM sessions written out by hand. ADVI then approximates the posterior over those parameters by a Gaussian with independent coordinates, chosen to be closest in KL divergence to the true posterior, by stochastic optimization of the evidence lower bound. State inference in the decision-support layer is the filtering distribution, the probability of each state given the observations so far, and it is computed per posterior draw, so the intervals around the 48-hour shock risks integrate over parameter uncertainty rather than plugging in a point estimate. Treatment effects are posterior quantities, not fitted constants.

Model A is not Bayesian. Expectation maximization returns one set of parameters with no prior and no posterior over them; its forward-backward pass gives state probabilities for fixed parameters, which is the same filtering idea without the uncertainty about the model.

What the project does not do is check the posterior against anything. There is no held-out likelihood for either model and no posterior predictive check. The convergence cell reports an effective sample size and R-hat for the ADVI draws, but those diagnostics are for Markov chains; draws from a fitted variational approximation are independent by construction, and the number only says how many were taken. Mean-field ADVI is also known to understate posterior spread, so the intervals in the reports are optimistic for that reason too.

## Results

All numbers are from the saved outputs of notebooks 05 and 06, on the same 1,502 patients the model was fit to.

**State profiles.** Standardized emission means for the features that separate the states most.

| Feature | Mild | Moderate | Shock |
|---|---|---|---|
| SOFA total | −0.61 | 0.20 | 1.32 |
| Creatinine | −0.42 | 0.02 | 0.99 |
| Bilirubin | −0.31 | −0.23 | 1.09 |
| Lactate | −0.32 | 0.06 | 0.95 |
| Platelets | 0.24 | −0.13 | −0.61 |
| Mean arterial pressure | 0.12 | −0.10 | −0.34 |

Heart rate, respiratory rate and the time gap barely differ between states.

**Treatment effects.** Sums of the posterior-mean transition coefficients toward more severe states and toward less severe states.

| Treatment | Toward worse states | Toward better states |
|---|---|---|
| Antibiotics | −0.82 | −0.28 |
| Vasopressors | +1.44 | −0.88 |
| IV fluids | −0.88 | +0.41 |

**Population counterfactual.** Starting from the fitted initial distribution and applying 48 hourly transitions under a fixed scenario, using posterior means.

| Scenario | Mild | Moderate | Shock |
|---|---|---|---|
| No treatment | 55.0% | 26.7% | 18.3% |
| Antibiotics only | 76.5% | 15.6% | 7.8% |
| Antibiotics and IV fluids | 85.8% | 10.3% | 3.8% |
| All three | 57.0% | 32.9% | 10.1% |

**Risk stratification.** Of the first 500 patients, 22 percent have a current shock probability above 30 percent. In-hospital mortality is 53.2 percent in that group and 8.2 percent in the rest, a ratio of 6.5.

**Patient reports.** The three illustrated patients sit at 100.0 percent in one state at every checkpoint of their stay. For the first, who died, the model says Mild throughout and recommends antibiotics with fluids, 2.4 percent shock risk against 17.7 percent untreated. For the third, in Shock at the end with a 26 percent chance of recovery untreated, no scenario moves that chance by more than 5 points. The delay sweep's "tipping point" lands at 1 hour for one patient and 22.5 hours for another.

**Model A.** No numbers were saved. Its one patient timeline shows the progression probability peaking near 14 percent, below the 30 percent alert line, through a stay of 150 hours.

## What to learn from this project

Course staff added this section. The Bayesian model is a real achievement for a course project; the points below are about what it is being asked to say.

- **Check whether a finding is built into your inputs.** The largest effect in the model is that vasopressors raise the odds of worsening. But being on vasopressors sets the cardiovascular component of SOFA to 3, SOFA is one of the emission features, and the Shock state is defined after the fact as the one with the highest SOFA mean. Part of that coefficient is the model recognizing its own input. Dropping SOFA from the emissions, or dropping the vasopressor component from SOFA, is the experiment that would separate the two.
- **Counterfactuals need support in the data.** Every patient in the cohort received antibiotics; that is how the cohort was defined. The "no treatment" scenario, and the finding that antibiotics cut 48-hour shock risk from 18 to 8 percent, compare hours on and off treatment within an already-treated group of patients who were, by and large, treated when they looked worse. The discussion names confounding by indication and applies it to vasopressors, but the same reasoning applies to the recommendations the system does make.
- **Validate out of sample, and at a time that matters.** All 1,502 patients are training data. The headline validation infers each patient's state at the end of the stay and compares it with death in hospital, but patients who die in the ICU are in their worst state at the end, so the 6.5-fold mortality ratio is close to circular. The useful test is a state inferred at a fixed early time, 24 hours say, on patients the model never saw, scored against what happened afterwards. The first patient illustrated died while the model called them Mild at every checkpoint, and the write-up does not remark on it.
- **Notice when your posterior is never uncertain.** Every state probability in the reports is 100.0 or 0.0 percent. Fourteen features treated as conditionally independent Gaussians, with scales learned per state, overcount the evidence at each observation, and across 170 observations the filtering distribution saturates. The credible intervals that remain come from parameter uncertainty alone, approximated by a mean-field method that understates it. A state posterior that is always certain is a modeling symptom, not a sign of a confident model.
- **Fit the method to the claim.** "Continuous-time HMM" names a discrete-time model whose transition logits take the time gap as a covariate, which is a sound idea but not a continuous-time Markov process. The training likelihood uses the treatment flags at the current observation while the decision-support code uses the previous one. The "tipping point" is the elbow of a nearly straight line, which is why it finds 1 hour for one patient and 22 hours for another.
- **Ship what you describe.** The write-up says Model A's transition and emission matrices were initialized from researched values; no committed code does that. Its model module was never pushed, its processing script is not valid Python, and a teammate had to re-implement it to run it. Two docs were left as templates with "XXXX" in them, and four files were one byte long. Say who did what: no author is attached to any component in the write-up.
- **Mind the data agreement.** No raw data was committed and the ignore file is right, but three MIMIC subject IDs reached public outputs, in a figure title, a printed table and another figure title. Strip identifiers from anything you save, and prefer indices or aggregate views in figures.

What the project does well: a real clinical question with a defensible, documented cohort definition and item lists; a PyMC HMM that most students would not attempt, with the forward recursion written out and masked for variable-length sequences, treatment-dependent transitions, and an honest ELBO trace; notebooks that explain each step before running it and keep their outputs, so the work can be read without data access; a decision-support layer that turns a posterior into something a clinician could look at; and a discussion that names the retrospective design and confounding by indication as limits.

## About this project

This project was built in Bayesian Machine Learning with Generative AI Applications (ADSP 32014), part of the Master of Science in Applied Data Science at the University of Chicago, in Winter 2026. It is shown here as an example for later cohorts.

## Changes by course staff

Course staff forked the authors' repository and made these edits. The authors' code is unchanged.

- Redacted three MIMIC-IV subject identifiers from saved outputs: painted over the ID in the title of `figures/Model A Output Example 2.png`, and cleared two outputs in `notebooks/exploratory_analysis.ipynb`, a five-row data preview and an autocorrelation figure whose title carried an ID. The code that produced them is unchanged.
- Removed four one-byte placeholder files (`src/model_2.py`, `src/train_model_2.py`, `figures/inferred_states_model1.png`, `figures/inferred_states_model2.png`) and two unfilled template documents (`docs/methodology.md`, `docs/results.md`) that referred to them.
- Renamed the authors' README to `REPORT.md`, unchanged, and added this README with the program banner.

## License

MIT, held by the authors. See [LICENSE](LICENSE). MIMIC-IV has its own terms, above.
