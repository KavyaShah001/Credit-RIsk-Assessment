# Publishing and demonstrating the project

## Create the destination

1. Sign in and open [GitHub's new repository page](https://github.com/new).
2. Choose your own account as owner and name the repository `credit-risk-assessment`.
3. Suggested description: `Financial statement analytics and bankruptcy-risk modeling with chronological validation, calibration, SQLite and a Streamlit dashboard.`
4. Select Public for a portfolio repository, or Private while preparing it.
5. Leave the README, .gitignore and license initialization options unselected. This project already has local Git history to upload.
6. Click **Create repository** and copy its HTTPS repository URL.

These steps follow the [official repository creation guide](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository).

## Upload the local commit

Open the existing project folder in your editor and use **Terminal → New Terminal**. Confirm that `git status` shows this project's files. After the local review commit exists, run the following, replacing `YOUR_USERNAME` with your GitHub username:

```bash
git remote add origin https://github.com/YOUR_USERNAME/credit-risk-assessment.git
git push -u origin main
```

Run `git remote -v` first if a remote might already exist. An existing correct `origin` does not need to be added again. Authenticate through your local Git credential flow if prompted; a GitHub account password is not accepted for HTTPS Git operations. See [GitHub authentication guidance](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/about-authentication-to-github).

## Check the published copy

- Confirm the README renders, including its architecture diagram.
- Open `src/credit_risk/models/` and confirm the Python model files are present.
- Read the generated results in `docs/temporal_results.md`.
- Check the **Actions** tab for the project checks. A configured workflow is not evidence of a successful hosted run until that run finishes.
- Follow the README reproduction commands from a fresh clone when checking another machine. The current local checks alone do not establish that a fresh hosted environment succeeds.

## Five-minute demonstration

1. Explain the reconstructed next-fiscal-year bankruptcy outcome and missing exact filing dates.
2. Select **Annual panel — chronological validation** in the dashboard.
3. Show one fiscal-year portfolio snapshot, then an anonymous company's statement history, ratios, PD and project grade.
4. Apply the operating-cost scenario. Explain the financial assumptions and show the model-generated change in PD.
5. Show the chronological split, unseen-company results, calibration and confusion matrix. Discuss missed events alongside ROC-AUC.
6. Download a company HTML report and explain how a canonical CSV can be scored.

Keep the Polish five-year experiment separate when comparing results. The two datasets have different outcome horizons and populations. Future model changes require a new evaluation plan; do not repeatedly optimize against the already inspected final test period.
