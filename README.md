# Zillow Regression


# Project Description

*Construct an ML Regression model that predicts propery tax assessed values ('taxvaluedollarcnt') of Single Family Properties using attributes of the properties.

*Find the key drivers of property value for single family properties. 

# Project Goal
* Find drivers for housing price on the Zillow 2017 dataset. Why are customers churning?
* Construct a ML classification model that accurately predicts customer churn
* Present your process and findings to the lead data scientist

# Steps to Reproduce
1) Clone this repo.
2) Acquire the data from Codeup's SQL server.
3) Put the data in the file containing the cloned repo.
4) Run notebook.

# Modules / Files

| Module/Filename        | Description                                                                 |
|:-----------------------|:----------------------------------------------------------------------------|
| wrangle.py             | Imports, scrubs, and splits the Zillow data.                                |
| test.py                | Contains functions to split and test data.                                  |
| zillow.csv             | Contains a small set of data for exploratory analysis.                      |
| all_maps.py            | Includes data and functions to map all 2017 listings.                       |
| evaluate.py            | Contains functions to evaluate model performance and metrics.               |
| importer.py            | Functions to pull in all data from the year 2017.                           |
| model.py               | Contains functions to create and evaluate different models.                 |
| module_test.py         | Creates test splits for model evaluation.                                   |
| README.md              | Provides an overview of the project.                                        |
| viz.py                 | A module dedicated to creating visualizations for data analysis.            |



# Data Dictionary
| Feature                    | Definition                                                             |
|:---------------------------|:-----------------------------------------------------------------------|
|id                          | Numeric. Unique identifier for each record/property in the dataset.    |
|parcelid                    | Numeric. Unique identifier for each parcel/land property.              |
|airconditioningtypeid       | Categorical. Identifies the type of air conditioning system in the property. |
|architecturalstyletypeid    | Categorical. Identifies the architectural style of the property.       |
|basementsqft                | Numeric. The total square footage of the basement area.                |
|bathroomcnt                 | Numeric. The total number of bathrooms in the property.                |
|bedroomcnt                  | Numeric. The total number of bedrooms in the property.                 |
|buildingclasstypeid         | Categorical. The building class type identifier.                       |
|buildingqualitytypeid       | Categorical. A measure of the building quality or condition.           |
|calculatedbathnbr           | Numeric. The calculated number of bathrooms in the property.           |
|decktypeid                  | Categorical. Identifies the type of deck present in the property.      |
|finishedfloor1squarefeet    | Numeric. The square footage of the first floor.                        |
|calculatedfinishedsquarefeet| Numeric. The calculated total square footage of the finished area.     |
|finishedsquarefeet12        | Numeric. The square footage of the finished area of the property.      |
|finishedsquarefeet13        | Numeric. The square footage of additional finished area not accounted in finishedsquarefeet12. |
|finishedsquarefeet15        | Numeric. Another measure of the square footage of the finished area.   |
|finishedsquarefeet50        | Numeric. The square footage of the property considering a different measurement approach.  |
|finishedsquarefeet6         | Numeric. Yet another measure of the square footage of the finished area.|
|fips                        | Categorical. Federal Information Processing Standards code for geographical regions. |
|fireplacecnt                | Numeric. The number of fireplaces in the property.                     |
|fullbathcnt                 | Numeric. The number of full bathrooms in the property.                 |
|garagecarcnt                | Numeric. The number of cars that can be accommodated in the garage.    |
|garagetotalsqft             | Numeric. The total square footage of the garage area.                  |
|hashottuborspa              | Bool. Indicates if the property has a hot tub or spa.                   |
|heatingorsystemtypeid       | Categorical. Identifies the type of heating system in the property.    |
|latitude                    | Numeric. The geographical latitude of the property.                    |
|longitude                   | Numeric. The geographical longitude of the property.                   |
|lotsizesquarefeet           | Numeric. The total square footage of the lot/land area.                |
|poolcnt                     | Numeric. The count of pools in the property.                           |
|poolsizesum                 | Numeric. The total size (sum of all pools) in the property.            |
|pooltypeid10                | Categorical. Identifier for a specific type of pool in the property.   |
|pooltypeid2                 | Categorical. Identifier for another specific type of pool in the property.|
|pooltypeid7                 | Categorical. Identifier for yet another specific type of pool in the property.|
|propertycountylandusecode   | Categorical. The land use code assigned by the county to the property. |
|propertylandusetypeid       | Categorical. Identifies the type of land use of the property.          |
|propertyzoningdesc          | Categorical. A description of the property zoning as defined by local authorities.|
|rawcensustractandblock      | Categorical. Raw data indicating the census tract and block of the property.|
|regionidcity                | Categorical. The identifier for the city where the property is located.|
|regionidcounty              | Categorical. The identifier for the county where the property is located.|
|regionidneighborhood        | Categorical. The identifier for the neighborhood where the property is located.|
|regionidzip                 | Categorical. The ZIP code where the property is located.               |
|roomcnt                     | Numeric. The total number of rooms in the property.                    |
|storytypeid                 | Categorical. Identifies the type of story structure of the property.   |
|threequarterbathnbr         | Numeric. The number of three-quarter bathrooms in the property.        |
|typeconstructiontypeid      | Categorical. Identifies the type of construction material used in the property. |
|unitcnt                     | Numeric. The number of units in the property.                          |
|yardbuildingsqft17          | Numeric. The square footage of yard buildings (other than the main property) present in the area.|
|yardbuildingsqft26          | Numeric. The square footage of additional yard buildings not accounted in yardbuildingsqft17.|
|yearbuilt                   | Numeric. The year the property was built.                              |
|numberofstories             | Numeric. The number of stories in the property.                        |
|fireplaceflag               | Bool. Indicates if the property has a fireplace.                       |
|structuretaxvaluedollarcnt  | Numeric. The assessed value of the structure for tax purposes.         |
|taxvaluedollarcnt           | Numeric. The total tax value of the property.                          |
|assessmentyear              | Numeric. The year the property was last assessed for tax purposes.     |
|landtaxvaluedollarcnt       | Numeric. The assessed value of the land for tax purposes.              |
|taxamount                   | Numeric. The total amount of tax levied on the property.               |
|taxdelinquencyflag          | Bool. Indicates if there is a tax delinquency on the property.         |
|taxdelinquencyyear          | Numeric. The year of tax delinquency if any.                           |
|censustractandblock         | Categorical. The census tract and block code where the property is located.|


# Initial Thoughts and Questions

*Why do some properties have a much higher value than others when they are located so close to each other?

*Why are some properties valued so differently from others when they have nearly the same physical attributes but only differ in location?

*Is having 1 bathroom worse for property value than having 2 bedrooms?


# Takeaways and Conclusions
* All elements of the MVP features effect value.
* Our model can predect the value of any house with an RSME of 562,913.96


# Recommendations
* We should focus on size, number of baths and number of bedrooms to predict price.
* I would like to get into the location data plot the lat longs.
* Next would be to fine tune models by going through hyperparameters.
* Limit data and train models on different location and price catagories.