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

Given that for a fact women in the LK discipline consistently perform higher scores than men and given the lack of data inputted data. I wondered if it was acceptable to aggregate the data over gender, especially so for the D2 and D3 categories as these were hard to properly fit to a statistical model.

To get a feel if my gut feeling that such aggregation is was tolerable, I turned to www.olympics.com and started to look at the result of the best of the world. Unfortunately I didn't immediately find the results of the main competition but only for the finals. This a elimination race of the 8 best competitors of the main competition. However the below table do show that women and men perform equally in these type of competitions.

![2020 Tokyo Olympic Finals](pictures/olympic_final.png?raw=true)

Although the used scripts can still work with gender specific categories, I believe in this use-case the better result is achieved be disregarding gender when trying to fit a statistical model to an age category.

## Distribution Fitting
## Normalization Factor
## Conclusion
## Results
## Cross Discipline
## Future Work
