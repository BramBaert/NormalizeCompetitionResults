# Introduction
For many sports it is common to have age categories to promote fair competition. However when competing within a local club more often than not, there are too few competitors within each category to actually speak of a competition.

With the lack of q sufficient basis of competitors, it is hard to motivation people to enter the competition and as such never obtaining critical mass to create a fun competition all together. This gave me the idea that we should get rid of the age categories however still maintain fair comparison amongst competitors while recognizing age difference does impact the performance within the sport.

Within this coding project I wanted to experiment if it is possible to create such a "fair" comparison across age groups by applying statistical method to determine correction factors to normalize the results over all competitors.

# Usage
1. Install python version 3.11 or above is advised
2. Install the required packages pip install -r requirements.txt

## Creation of your own correction factors
CLI to be made

## Normalizing results
CLI to be made

# Process
This chapter describes the process I went through and as such what is finally used to obtain the said objective.
## Context
This project focuses on target shooting competitions where each competitor takes a number of shots on a target. For each shot points are awarded based on the shots position w.r.t. the center of the target. All shots are then cumulated to get the final competition result. 

To obtain a personal benefit from the project I am focusing on the ISSF 10m Air Rifle and 10m Air Pistol disciplines within the scope of Belgium. From now on these will be referenced as LK and LP respectively being the abbreviation of the Dutch Luchtkarabijn and Luchtpistool. Within Belgium we have the following age categories:

|  Category   | Abbreviation |   Age   | Shots per Competition |
| :---------: | :----------: | :-----: | :-------------------: |
|  Duiveltje  |     DUV      |  < 12   |          30           |
|  Benjamin   |     BEN      | 12 - 13 |          30           |
|    Cadet    |     CAD      | 14 - 15 |          40           |
| Junior Dame |      JD      | 16 - 20 |          60           |
| Junior Heer |      JH      | 16 - 20 |          60           |
|   Dame 1    |      D1      | 21 - 49 |          60           |
| Senioren 1  |      S1      | 21 - 49 |          60           |
|   Dame 2    |      D2      | 50 - 59 |          60           |
| Senioren 2  |      S2      | 50 - 59 |          60           |
|   Dame 3    |      D3      |  > 60   |          60           |
| Senioren 3  |      S3      |  > 60   |          60           |

## Normalizing for gender
Historically there were difference between to rules for women and men, more specifically women only had to perform 40 shots in a single competition while this was 60 for men. However for many years now this have been aligned with the 60 shots applicable for men. 

Given that for a fact women in the LK discipline in Belgium consistently perform higher scores than men and given the lack of data inputted data. I wondered if it was acceptable to aggregate the data over gender, especially so for the D2 and D3 categories as these were hard to properly fit to a statistical model.

To get a feel if my gut feeling that such aggregation is was tolerable, I turned to www.olympics.com and started to look at the result of the best of the world. Unfortunately I didn't immediately find the results of the main competition but only for the finals. This a elimination race of the 8 best competitors of the main competition. However the below table does show that women and men perform equally in these type of competitions.

![2020 Tokyo Olympic Finals](pictures/olympic_final.png?raw=true)

Although the used scripts can still work with gender specific categories, I believe in this use-case the better result is achieved by disregarding gender when trying to fit a statistical model to an age category.

## Distribution Fitting
My initial thought process was to try and fit a skewed distribution to the data however this proved to result in rather poor results. After doing more research on possible distribution functions I came across http://zoonek2.free.fr/UNIX/48_R/07.html where I picked up the beta function as a candidate. It looked promising because of its feature of being able to create a sharp fall-off at the end of the distribution, as can be seen below. This really applicable to point based competition result with a maximum achievable result.
![Theoretical beta functions](pictures/beta_function.png?raw=true)
On my initial data set containing 3 years worth of national competition results  this still didn't get me the desired performance. Increasing the dataset to 5 years worth of data did improve the results but left some categories that really didn't behave at all

Looking around further, I came across the python lib distfit (https://erdogant.github.io/distfit/pages/html/index.html). This library enables to automate the process of testing the performance in fitting of different distribution algorithms. After shaping the data set and some more scripting I got the following output.
![Fitting results for different types of distributions](pictures/dist_fitting_unisex.png?raw=true)
It shows a sorted result of the different distribution algorithms based on the cumulative score for all categories. With this result I added loggamma and genextreme to my analysis. This results in the below graphs.
![Probability Density functions](pictures/pdf_unisex.png?raw=true)
Note the strange behavior of the beta PDF function for the 'CAD' category. This is also the case for the 'D2' category when not combining genders.

To be able to analyse the results the below picture gives an idea how things look when you have an ideal data set. Notice that the line of the distribution function crosses the center of each bar.

![Ideal histogram](pictures/ideal_histogram.jpg?raw=true)

Doing this analysis for our data, it is my opinion that the loggamma function is better at capturing the peak and the sharp fall-off at the end. This is then also the distribution function I used for the following chapters.

## Normalization
Now that we have the parameters that describe how competitors within a category are distributed we have to find a way to translate that to a "reference". The first idea was to use the linear spacing between the 1% and 99% cumulative density function points (points = the result achieved in the competition) for each distribution. 

To put it in a formula the linear spacing can be defined as following: 
> **LinSpace**<sub>(CDF<sub>cat</sub>,n)</sub> = CDF<sub>cat</sub><sub>(.01)</sub> + ( ( CDF<sub>cat</sub><sub>(.99)</sub>  – CDF<sub>cat</sub><sub>(.01)</sub> ) / <*table_size*> ) * n

This same linear distance in the correction table can then be used to determine the score in the "reference" category. However when I reviewed the normalized results, the outcome did not match my expectations.


Thinking further about it I realized that the linear spacing isn't the proper way of doing it as the slope acuteness for the respective cumulative density functions differs between categories. This means that from one category the probability of a point being achieved between two linearly spaced points is different. To account for this a second method is used that finds the point in the "reference" category that has a matching cumulative density. With the point from the source category and the reference category a multiplier can then be found to normalize the competitors result.

The below figure gives a visual aid on how each compensation method works.
![Normalization methods](pictures/normalization_methods.png?raw=true)
### Normalization for club competitions
Above we described methods on how to normalize results from competitions following the national rules. However often club and regional competitions adopt a short format for their competitions to lower the barrier for entry. 

This means that we need to calculate compensation factors in function of those shorter format competitions, practically for this project this means a lower shot count. This is achieved be pre-processing the input data and normalize the shot count based on an input parameter "competition shot count" and the shot count per category as defined in one of the previous chapters.
## Results
### Background on box plots
To analyse the results of the normalization process we 'll make use of boxplots. The below graph gives an explanation on how to read these.
![Box plot of the normalized input data](pictures/reading_boxplots.png?raw=true)
What it comes down to is the solid lines of the box contain the center most half of all values in the data set with the solid line in the middle being the exact half point (median). The dashed line is the average (mean). The "whiskers" indicate the top and bottom 25% of data points respectively.

If the normalization process works perfectly each of these key features (min, q1,median, q3 and max) should align with the reference category.
### Results in a graph
Below is a box plot graph of the normalization process applied to the input data itself.
![Box plot of the normalized input data](pictures/result_box_plot_unisex.png?raw=true)
The reason why I came up with the second method for normalization (the one base on the CDF value). Is the difference in the q1, median and q3 features for the youth categories.

### Results in a ranking
Just for the gist of it, applying this method to the results of the latest national competitions give a TOP 25 raking as following

![BOA 2023 Top 25 ranking post normalization](pictures/BOA_2023_Top25.png?raw=true)

## Conclusion
2 Methods have been provided to normalize the points of a skill based result competition over different age categories. Both methods provide a decent result with the LinSpace method giving the better result in our practical example. The CDF method is however mathematical the better one.
## Cross Discipline
Although currently not provided the process can also be applied across disciplines. 
## Future Work
- In 2024 the Junior categories has been changed to go from 16 - 17 years old to 16 - 20 years old. As the data we have still works with the previous age group, we could check Junior competitor names across the different years and alter the data to post-pone there respective category change to D1 / S1 with 3 years. Ideally this would be done on 8 years of data and then drop the earliest 3 years to maintain sufficient "corrected" data for proper fitting. 
- Create a CLI interface to have a parametrically way of fitting data and creating the correction tables.
- Create a CLI interface to use pre-existing correction tables to normalize results provided in a CSV file selected by input arguments
- Create a example calculation sheet (libre office, Excel, Google Sheets) that can use a correction table to normalize results directly in a printable format for club competitions.
- Adding the LP discipline data
