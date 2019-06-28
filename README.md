# CareFinder
## Custom Data-Driven Hospital Choices in your Neighborhood

### App Overview
For my Insight (Health Data Science; 2019B) project, I built an app that allows users to search for hospitals in their neighborhood filtered by their department of choice and ranked by user provided weights for various factors. The app provides a ranked choice of the hospital options in their neighborhood, their locations on a map and a chart comparing the top 5 hospitals on the various metrics selected. The app can be accessed at https://vs-hosp-compare.herokuapp.com

### Code Overview
The file "location_df_script.ipynb" is a Jupyter Notebook that was originally used to pull together data from various sources in order to build the "master_df" that was used as the database for the app. It also includes some EDA and preliminary attempts made to predict the level of satisfaction patients had with each hospital based on survey responses. "Week3_work.ipynb" includes code to separate hospitals by departments and continues building features that were used in the final app. "app2.py" is the file that is used to build the app (using Dash hosted on Heroku).

The Data folder contains most of the data sources used to build the product except for one large file. The "credentials" folder contains a Mapbox key that was used for an earlier version of this project that has since been updated.

### License
This app has been licensed with the GNU General Public License v3.0. The full license is available under "License.txt".

### Disclaimer
This project was built over the course of 3 weeks and is used primarily as a proof of concept and for demonstration purposes. The full disclaimer can be found in "Disclaimer/discl.py".
