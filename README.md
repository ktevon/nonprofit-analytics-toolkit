🧰 **Nonprofit Analytics Toolkit**

_An evolving toolkit designed to support data-informed decision-making in the nonprofit sector._

📌 **Overview**

While nonprofit organisations generate rich, complex data, real datasets are often sensitive, inconsistent, or simply unavailable. The **Nonprofit Analytics Toolkit** aims to make exploration easier by providing:

- Synthetic datasets that mimic real-world nonprofit behaviour
- Reproducible code for generating and analysing charity data
- A foundation for modelling, benchmarking, and scenario testing in the nonprofit sector

This repository is in its early stages and will expand over time as new modules, datasets, and analytical workflows are added.

📂 **Current Contents**

**animal_charity_donation_hierarchical_clustering.py**
A Python script demonstrating hierarchical clustering on mixed‑type donor data using the publicly available animal_charity_donation_records.csv dataset from Kaggle. The workflow calculates Gower distance to accommodate categorical and numeric variables, performs hierarchical clustering, evaluates candidate cluster solutions using silhouette analysis, and visualises the resulting segments with a radar chart.

This script serves as a practical example of donor segmentation using mixed‑type data — a common scenario in nonprofit analytics.

**giw_propensity_assign_bq_status.py**
**giw_propensity_modelling.py**

Python scripts that use the synthetic charity dataset (see **synthetic_charity_dataset_fix_attempt3.py** below) to build a classification model identifying donors who are more likely to leave a gift in their will. This analysis is documented in forthcoming Medium articles.

**salesforce_rfm_segmentation_pipeline.r**

A fully anonymised R script demonstrating how to automate an RFM segmentation workflow using Salesforce data. The script retrieves supporter and transaction data via the Salesforce Bulk 1.0 API, calculates RFM scores, applies k‑means clustering to generate donor segments, and writes the resulting segment labels back to Salesforce. All sensitive information (credentials, endpoints, and IDs) has been removed and/or replaced with placeholders.

Useful as a reference implementation for nonprofits looking to operationalise donor segmentation using open‑source tools.

**synthetic_charity_dataset_fix_attempt3.py**

This script generates a synthetic charity dataset, designed to resemble realistic patterns in nonprofit operations.

It includes logic for modelling features such as:

- Organisational size
- Revenue and fundraising patterns
- Program activity
- Donor behaviour
- Demographic distribution

The dataset is designed to be:

- **Realistic enough** for meaningful analysis
- **Fully synthetic**, containing no sensitive or identifiable information
- **Customisable**, allowing future iterations to incorporate additional features or sector nuances

It was built through multiple interations--**synthetic_charity_dataset.py**, **synthetic_charity_dataset_fix_attempt1.py**, and **synthetic_charity_dataset_fix_attempt2.py**. This process is fully documented in the following Medium articles.

- https://medium.com/data-and-beyond/how-i-built-a-synthetic-charity-dataset-that-behaves-like-the-real-thing-f19af0cf548d
- https://medium.com/data-and-beyond/the-day-my-synthetic-donors-didnt-pass-for-human-e67fb52f928f

🎯 **Purpose and Use Cases**

This toolkit supports a range of analyitcs activities commonly encountered in the NFP sector, including:

- Exploratory data analysis for nonprofit organisations
- Benchmarking and performance modelling
- Scenario simulations (e.g., funding shocks, donor churn)
- Machine learning examples using synthetic data
- Reproducible workflows for analysts and students

🚀 **Getting Started**

To generate the synthetic dataset:

  synthetic_charity_dataset_fix_attempt3.py

This will produce a CSV (or other output, depending on your script) containing the synthetic charity records.

🛠️ **Roadmap**

Planned enhancements include:

- Additional synthetic datasets (e.g., volunteers, programs)
- Data validation and profiling tools
- Example notebooks for analysis and visualisation
- Documentation on methodology and assumptions
- Packaging the toolkit for easier installation

🤝 **Contributing**

Contributions, suggestions, and issue reports are welcome.

As the project matures, contribution guidelines will be added.

📄 **License**

MIT
