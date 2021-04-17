#Final project 
library('tidyverse') #librarying tidyverse in order to work with piped functions later
library('RCurl')
library('jsonlite')
library('stringr') 
library('ggplot2')
airlineDF<-jsonlite::fromJSON('fall2019-survey-M02(1).json')
summary(airlineDF)
str(airlineDF)
sum(is.na(airlineDF))
head(airlineDF)
#checking for missing values in each column 
sum(is.na(airlineDF$Destination.City))
sum(is.na(airlineDF$Origin.City))
sum(is.na(airlineDF$Airline.Status))
sum(is.na(airlineDF$Age))
sum(is.na(airlineDF$Gender))
sum(is.na(airlineDF$Price.Sensitivity))
sum(is.na(airlineDF$Year.of.First.Flight))
sum(is.na(airlineDF$Flights.Per.Year))
sum(is.na(airlineDF$Loyalty))
sum(is.na(airlineDF$Type.of.Travel))
sum(is.na(airlineDF$Total.Freq.Flyer.Accts))
sum(is.na(airlineDF$Shopping.Amount.at.Airport))
sum(is.na(airlineDF$Eating.and.Drinking.at.Airport))
sum(is.na(airlineDF$Class))
sum(is.na(airlineDF$Day.of.Month))
sum(is.na(airlineDF$Flight.date))
sum(is.na(airlineDF$Partner.Code))
sum(is.na(airlineDF$Partner.Name))
sum(is.na(airlineDF$Origin.State))
sum(is.na(airlineDF$Destination.State))
sum(is.na(airlineDF$Scheduled.Departure.Hour))
sum(is.na(airlineDF$Departure.Delay.in.Minutes))#208 missing 
sum(is.na(airlineDF$Flight.cancelled))
sum(is.na(airlineDF$Flight.time.in.minutes)) #238 missing 
sum(is.na(airlineDF$Flight.Distance))
sum(is.na(airlineDF$Likelihood.to.recommend)) #1 missing 
sum(is.na(airlineDF$olong))
sum(is.na(airlineDF$olat))
sum(is.na(airlineDF$dlong))
sum(is.na(airlineDF$dlat))
sum(is.na(airlineDF$freeText)) #10000 missing
hist(airlineDF$Departure.Delay.in.Minutes[airlineDF$Departure.Delay.in.Minutes<150],)#right skewed distribution 
#therefore median is more appropriate replacement value for na
hist(airlineDF$Flight.time.in.minutes[airlineDF$Departure.Delay.in.Minutes],)
med_flight<-median(airlineDF$Flight.time.in.minutes, na.rm=TRUE)
med_dep<-median(airlineDF$Departure.Delay.in.Minutes, na.rm=TRUE) #median is equal to 0 
airlineDF$Departure.Delay.in.Minutes[is.na(airlineDF$Departure.Delay.in.Minutes)]<-med_dep
airlineDF$Flight.time.in.minutes[is.na(airlineDF$Flight.time.in.minutes)]<-med_flight
sum(is.na(airlineDF$Departure.Delay.in.Minutes)) #0 remain
sum(is.na(airlineDF$Flight.time.in.minutes)) #0 remain 
mean_like<-mean(airlineDF$Likelihood.to.recommend, na.rm=TRUE)
airlineDF$Likelihood.to.recommend[is.na(airlineDF$Likelihood.to.recommend)]<-mean_like
sum(is.na(airlineDF$Likelihood.to.recommend)) #0 remain 
########################################################Exploratory Data Analysis
View(airlineDF)

averageClassRating<-airlineDF %>% 
  group_by(Class) %>% 
  summarise(average = mean(x = Likelihood.to.recommend))
averageClassRating %>% arrange(desc(average))

averagePartnerRating<-airlineDF %>% 
  group_by(Partner.Name) %>% 
  summarise(average = mean(x = Likelihood.to.recommend))
averagePartnerRating %>% arrange(desc(average))


hist(airlineDF$Likelihood.to.recommend[airlineDF$Likelihood.to.recommend], breaks = 8)

airlineDF$NPS<-cut(airlineDF$Likelihood.to.recommend, breaks = c(0,6,8,10), labels=c("Detractor","Passive","Promoter"))
View(airlineDF)
count<-table(airlineDF$NPS)



ggplot(airlineDF, aes(x = NPS, y = ..count..)) + geom_bar(position = "dodge")


airlineDF$AgeGroup<-cut(airlineDF$Age, breaks = c(10,25,40,63,85), labels=c("Teenager","Young Adults","Middle Aged","Older"))
View(airlineDF)

hist(airlineDF$Age, breaks = 30)
min(airlineDF$Age)
max(airlineDF$Age)
summary(airlineDF$Age)


ggplot(airlineDF, aes(x = airlineDF$FlightDist, y = ..count..)) + geom_bar(aes(fill=NPS=="Promoter"),position = "identity")
#teens comprise very small portion of NPS so won't include age group in model 
#work on age groups 
table(airlineDF$Class, airlineDF$Likelihood.to.recommend)
summary(airlineDF$Flight.Distance)

airlineDF$FlightDist<-cut(airlineDF$Flight.Distance, breaks = c(50,450,700,1400,4000), labels=c("Short","Medium","Longer Aged","Long"))






