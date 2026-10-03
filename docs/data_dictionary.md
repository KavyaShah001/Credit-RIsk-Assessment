# Data dictionary — UCI Polish Companies Bankruptcy, 1st-year case

## Source and target

- Source: [UCI Machine Learning Repository, Polish Companies Bankruptcy](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data)
- Citation: Tomczak, S. (2016). *Polish Companies Bankruptcy* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5F600.
- License: Creative Commons Attribution 4.0 International (CC BY 4.0). Attribution must accompany redistributed data or adaptations.
- Selected file: `1year.arff`, converted locally to `data/raw/polish_bankruptcy_1year.csv`.
- Prediction target: the project's binary `defaulted` field maps the source `class`: 1 indicates bankruptcy in the five-year outcome window and 0 indicates no bankruptcy in that window. Bankruptcy is the observed outcome and is used as a proxy for default risk; it is not a direct record of missed payments or a lender's default definition.
- Source fields: 64 precomputed financial ratios, not raw financial statement line items. The source has no persistent company identifier, company name, sector, exposure, or LGD field in the selected file.
- Source-period limitation: UCI reports bankrupt observations from 2000–2012 and still-operating observations from 2007–2013. The different class observation windows may introduce period effects; the dataset does not provide the identifiers and dates needed to fully control for them.
- Observation identity: `UCI-1YEAR-######` is an internally generated row key only; it does not identify a real company.
- Time split limitation: the selected file does not provide a reporting date or stable company identifier. A company-grouped or time-aware holdout cannot be established from this file. Any split design must state this limitation.
- Missingness: UCI marks the dataset as containing missing values. The ingestion pipeline preserves missing ratio values as SQL `NULL`; it does not replace them with zero.
- Ingestion snapshot: 7,027 rows; 271 defaults (3.86%); 5,835 missing cells across the 64 ratio columns. The source CSV is a fixed snapshot in `data/raw/` and is ignored by Git.

## Ratio fields

Definitions below follow the UCI variable information. These are source-supplied ratios and should not be confused with calculations from full raw statements. Category assignment is a presentation aid; interpretation depends on accounting context and sector.

| CSV field | UCI field | Category | Definition |
|---|---|---|---|
| `ratio_01` | Attr1 | Profitability | Net profit / total assets |
| `ratio_02` | Attr2 | Leverage / solvency | Total liabilities / total assets |
| `ratio_03` | Attr3 | Liquidity / working capital | Working capital / total assets |
| `ratio_04` | Attr4 | Liquidity / working capital | Current assets / short-term liabilities |
| `ratio_05` | Attr5 | Liquidity / working capital | (Cash + short-term securities + receivables - short-term liabilities) / (operating expenses - depreciation) × 365 |
| `ratio_06` | Attr6 | Profitability | Retained earnings / total assets |
| `ratio_07` | Attr7 | Profitability | EBIT / total assets |
| `ratio_08` | Attr8 | Leverage / solvency | Book value of equity / total liabilities |
| `ratio_09` | Attr9 | Efficiency / other | Sales / total assets |
| `ratio_10` | Attr10 | Leverage / solvency | Equity / total assets |
| `ratio_11` | Attr11 | Profitability | (Gross profit + extraordinary items + financial expenses) / total assets |
| `ratio_12` | Attr12 | Profitability | Gross profit / short-term liabilities |
| `ratio_13` | Attr13 | Profitability | (Gross profit + depreciation) / sales |
| `ratio_14` | Attr14 | Profitability | (Gross profit + interest) / total assets |
| `ratio_15` | Attr15 | Leverage / solvency | (Total liabilities × 365) / (gross profit + depreciation) |
| `ratio_16` | Attr16 | Cash-flow / earnings proxy | (Gross profit + depreciation) / total liabilities |
| `ratio_17` | Attr17 | Leverage / solvency | Total assets / total liabilities |
| `ratio_18` | Attr18 | Profitability | Gross profit / total assets |
| `ratio_19` | Attr19 | Profitability | Gross profit / sales |
| `ratio_20` | Attr20 | Efficiency / other | Inventory × 365 / sales |
| `ratio_21` | Attr21 | Efficiency / other | Sales in year n / sales in year n-1 |
| `ratio_22` | Attr22 | Profitability | Profit on operating activities / total assets |
| `ratio_23` | Attr23 | Profitability | Net profit / sales |
| `ratio_24` | Attr24 | Profitability | Gross profit (in 3 years) / total assets |
| `ratio_25` | Attr25 | Leverage / solvency | (Equity - share capital) / total assets |
| `ratio_26` | Attr26 | Cash-flow / earnings proxy | (Net profit + depreciation) / total liabilities |
| `ratio_27` | Attr27 | Coverage / obligations | Profit on operating activities / financial expenses |
| `ratio_28` | Attr28 | Liquidity / working capital | Working capital / fixed assets |
| `ratio_29` | Attr29 | Efficiency / other | Logarithm of total assets |
| `ratio_30` | Attr30 | Leverage / solvency | (Total liabilities - cash) / sales |
| `ratio_31` | Attr31 | Profitability | (Gross profit + interest) / sales |
| `ratio_32` | Attr32 | Efficiency / other | Current liabilities × 365 / cost of products sold |
| `ratio_33` | Attr33 | Coverage / obligations | Operating expenses / short-term liabilities |
| `ratio_34` | Attr34 | Coverage / obligations | Operating expenses / total liabilities |
| `ratio_35` | Attr35 | Profitability | Profit on sales / total assets |
| `ratio_36` | Attr36 | Efficiency / other | Total sales / total assets |
| `ratio_37` | Attr37 | Liquidity / working capital | (Current assets - inventories) / long-term liabilities |
| `ratio_38` | Attr38 | Leverage / solvency | Constant capital / total assets |
| `ratio_39` | Attr39 | Profitability | Profit on sales / sales |
| `ratio_40` | Attr40 | Liquidity / working capital | (Current assets - inventory - receivables) / short-term liabilities |
| `ratio_41` | Attr41 | Leverage / solvency | Total liabilities / ((profit on operating activities + depreciation) × (12/365)) |
| `ratio_42` | Attr42 | Profitability | Profit on operating activities / sales |
| `ratio_43` | Attr43 | Efficiency / other | Receivables plus inventory turnover in days |
| `ratio_44` | Attr44 | Efficiency / other | Receivables × 365 / sales |
| `ratio_45` | Attr45 | Efficiency / other | Net profit / inventory |
| `ratio_46` | Attr46 | Liquidity / working capital | (Current assets - inventory) / short-term liabilities |
| `ratio_47` | Attr47 | Efficiency / other | Inventory × 365 / cost of products sold |
| `ratio_48` | Attr48 | Cash-flow / earnings proxy | Dataset's EBITDA / total assets measure |
| `ratio_49` | Attr49 | Cash-flow / earnings proxy | Dataset's EBITDA / sales measure |
| `ratio_50` | Attr50 | Liquidity / working capital | Current assets / total liabilities |
| `ratio_51` | Attr51 | Leverage / solvency | Short-term liabilities / total assets |
| `ratio_52` | Attr52 | Efficiency / other | Short-term liabilities × 365 / cost of products sold |
| `ratio_53` | Attr53 | Leverage / solvency | Equity / fixed assets |
| `ratio_54` | Attr54 | Leverage / solvency | Constant capital / fixed assets |
| `ratio_55` | Attr55 | Liquidity / working capital | Working capital |
| `ratio_56` | Attr56 | Profitability | (Sales - cost of products sold) / sales |
| `ratio_57` | Attr57 | Liquidity / working capital | (Current assets - inventory - short-term liabilities) / (sales - gross profit - depreciation) |
| `ratio_58` | Attr58 | Profitability | Total costs / total sales |
| `ratio_59` | Attr59 | Leverage / solvency | Long-term liabilities / equity |
| `ratio_60` | Attr60 | Efficiency / other | Sales / inventory |
| `ratio_61` | Attr61 | Efficiency / other | Sales / receivables |
| `ratio_62` | Attr62 | Efficiency / other | Short-term liabilities × 365 / sales |
| `ratio_63` | Attr63 | Efficiency / other | Sales / short-term liabilities |
| `ratio_64` | Attr64 | Efficiency / other | Sales / fixed assets |

The precise wording of some source formulas (notably its EBITDA measures) should be checked against the dataset paper before detailed accounting interpretation. The MVP will use these fields as documented model inputs rather than reconstructing the underlying amounts.
