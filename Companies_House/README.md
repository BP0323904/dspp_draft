## Question:
####  To what extent can company structure and ownership variables be used to predict late filing behaviour in UK companies, using logistic regression, as an indicator of regulatory non-compliance?

##### The main discussion and findings are discussed in a written report which is not stored on this github reporsitory. However the code used for the analysis can be found [here](https://github.com/BP0323904/dspp/tree/main/Companies_House/Notebooks) and [here](https://github.com/BP0323904/dspp/tree/main/Companies_House/src).

I took BasicCompanyDataAsOneFile-2026-03-02.csv data from [Companies House - Free Company Data Product Data Source](https://download.companieshouse.gov.uk/en_output.html) and persons-with-significant-control-snapshot-2026-03-12.txt data from [Companies House - People with significant control (PSC) snapshot Data Source](https://download.companieshouse.gov.uk/en_pscdata.html) to conduct logistic regression modelling on company late-filing behaviours.


### Data:
The final modelling dataset consisted of the following...

| variable                 | class    | description                                                                                                                                      |
|:-------------------------|:---------|:-------------------------------------------------------------------------------------------------------------------------------------------------|
| company_category         | category | Feature engineered variable, mapping of *company_category* into fewer categories.                                                                |
| registered_country       | category | Feature engineered variable, mapping of *registered_country* into fewer categories.                                                              |
| industry                 | category | Feature engineered variable, mapping of sic codes into fewer categories.                                                                         |
| company_age_when_acc_due | category | Feature engineered variable, binning of calculated ages into fewer categories, original data used = 'incorporation_date' and 'account_due_date'. |
| has_any_psc              | int8     | Binary variable created from feature generated *psc_count* column of aggregated data from PSC dataset.                                           |
| has_corporate_psc        | in8      | Binary variable, created from aggregating 'kind' variable in PSC dataset.                                                                        |
| recent_psc_change        | in8      | Binary variable, created from aggregating 'motified_on' and 'ceased_on' variables from PSC datast.                                               |
| overdue        	         | in8      | Binary target variable, created using 'account_due_date'.                                                                                        |
