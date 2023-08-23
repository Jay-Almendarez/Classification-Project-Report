# Classification-Project-Report
---
## Primary Goals: 
- Determine key drivers of churn from Telco Dataset. 
- Determine the most viable ML model and accurately predict churn with it.
---
## Procedure
- Acquire telco dataset from MySQLWorkbench
- prepare data by removing excessive features
---
## Data Dictionary
| Feature | Definition | 
| :- | :- |
| Senior Citizen | If a customer is a senior citizen, 0 = No, 1 = Yes |
| Tenure | The amount of months a customer has been with or is currently with company |
| <font color='red'>Monthly Charges</font> | Amount a customer is charged monthly |
| Total Charges | Cumulative amount a customer has paid |
| Gender | If a customer is male or female, 0 = Female, 1 = Male |
| Has Partner | If a customer has a partner, 0 = No, 1 = Yes |
| Has Dependents | If a customer has dependents, 0 = No, 1 = Yes |
| Has Multiple Lines | If a customer has multiple lines, 0 = No, 1 = Yes |
| <font color='red'>Contract</font> | Type of contract customer has, 0 = Month-to-month, 1 = One year, 2 = Two year|
| Internet Service | Type of Internet Service customer has, 0 = No internet service, 1 = DSL, 2 = Fiber optic |
| Has Automatic Payment | If a customer has automatic payment, 0 = No, 1 = Yes |
| Has Amenities | If a customer has a majority of amenities from (tech_support, online_security, paperless_billing, streaming_movies, online_backup, streaming_tv, device_protection), 0 = No, 1 = Yes |
| <font color='red'>Has Internet Service</font> | If a customer has internet service, 0 = No, 1 = Yes |
| Churn (Target) | If a customer has churned, False = No, True = Yes |
---
