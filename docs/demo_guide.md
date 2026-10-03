# Demonstrating the project

## Start the application

Follow the reproduction commands in the [README](../README.md#reproduce-the-annual-experiment) to prepare the source, train the model and create the SQLite database. From the project directory, activate the environment and launch the dashboard:

```bash
source .venv/bin/activate
streamlit run dashboard/app.py
```

Select **Annual panel — chronological validation** in the sidebar.

## Five-minute walkthrough

1. Explain the reconstructed next-fiscal-year bankruptcy outcome and missing exact filing dates.
2. Show one fiscal-year portfolio snapshot and its PD and project-grade distributions.
3. Open an anonymous company's statement history, financial ratios, PD, project grade and feature sensitivities.
4. Apply the operating-cost scenario. Explain the assumptions and show the model-generated change in PD.
5. Show the chronological split, unseen-company results, calibration and confusion matrix. Discuss missed events alongside ROC-AUC.
6. Download a company HTML report and explain how a canonical CSV can be scored.

Keep the Polish five-year experiment separate when comparing results. The two datasets have different outcome horizons and populations.

## Explain the evidence

Use [generated chronological results](temporal_results.md) for the actual numbers and the [interview guide](interview_guide.md) for the rationale. Feature sensitivities describe model associations, and scenario changes are conditional on their accounting assumptions. Neither establishes a causal effect.

The source contains neither exposure at default nor loss given default. Portfolio summaries therefore describe company counts and modeled PD distributions, not exposure-weighted expected loss.

## Repository maintenance

GitHub's **Actions** tab shows the existing unit tests and dependency consistency checks for each update. These software checks do not replace the full historical model evaluation. Reproduction on another machine requires the data-preparation and training commands in the README.

Future modeling changes need a new evaluation plan; repeatedly optimizing against the already inspected final test period would compromise its role as a holdout.
