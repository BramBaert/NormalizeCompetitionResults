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

To obtain a personal benefit from the project I am focusing on the ISSF 10m Air Rifle and 10m Air Pistol disciplines within the scope of Belgium. Within Belgium we have the following age categories.

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
## Distribution Fitting
## Normalization Factor
## Conclusion
## Results
## Cross Discipline
## Future Work
