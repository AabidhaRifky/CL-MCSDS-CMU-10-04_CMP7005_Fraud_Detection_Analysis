# Data Dictionary

| Column         | Data_Type   |   Missing_Values |   Unique_Values | Example_Values                                                     |
|:---------------|:------------|-----------------:|----------------:|:-------------------------------------------------------------------|
| ID             | int64       |                0 |           25134 | 5008806, 5008808, 5008809                                          |
| GENDER         | object      |                0 |               2 | M, F                                                               |
| CAR            | object      |                0 |               2 | Y, N                                                               |
| REALITY        | object      |                0 |               2 | Y, N                                                               |
| NO_OF_CHILD    | int64       |                0 |               9 | 0, 3, 1                                                            |
| FAMILY_TYPE    | object      |                0 |               5 | Married, Single / not married, Civil marriage                      |
| HOUSE_TYPE     | object      |                0 |               6 | House / apartment, Rented apartment, Municipal apartment           |
| FLAG_MOBIL     | int64       |                0 |               1 | 1                                                                  |
| WORK_PHONE     | int64       |                0 |               2 | 0, 1                                                               |
| PHONE          | int64       |                0 |               2 | 0, 1                                                               |
| E_MAIL         | int64       |                0 |               2 | 0, 1                                                               |
| FAMILY SIZE    | float64     |                1 |              11 | 2.0, 1.0, 5.0                                                      |
| BEGIN_MONTH    | int64       |                0 |              61 | 29, 4, 26                                                          |
| AGE            | int64       |                0 |              50 | 59, 52, 46                                                         |
| YEARS_EMPLOYED | float64     |                9 |              44 | 3.0, 8.0, 2.0                                                      |
| TARGET         | int64       |                0 |               2 | 0, 1                                                               |
| INCOME         | float64     |                0 |             195 | 112500.0, 270000.0, 135000.0                                       |
| INCOME_TYPE    | object      |               12 |               6 | Working, Commercial associate, State servant                       |
| EDUCATION_TYPE | object      |                0 |               5 | Secondary / secondary special, Higher education, Incomplete higher |