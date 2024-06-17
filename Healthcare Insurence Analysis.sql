create database healthcare;
use healthcare;
show databases;

SELECT * FROM healthcare.hospitalisation_details;
SELECT * FROM healthcare.medical_examinations;

/* 1. To gain a comprehensive understanding of the factors influencing hospitalization costs, it
is necessary to combine the tables provided. Merge the two tables by first identifying the
columns in the data tables that will help you in merging.*/

# Joining two tables

SELECT *
FROM healthcare.hospitalisation_details
JOIN healthcare.medical_examinations
ON hospitalisation_details.Customer_Id = medical_examinations.Customer_Id;

/* 2. Retrieve information about people who are diabetic and have heart problems with their
average age, the average number of dependent children, average BMI, and average
hospitalization costs */

SELECT HBA1C,`Heart Issues`, AVG(children), AVG(BMI), AVG(charges) 
FROM healthcare.medical_examinations AS m INNER JOIN healthcare.hospitalisation_details
AS h ON h.`Customer_Id` = m.`Customer_Id` WHERE m.HBA1C>6.5 AND m.`Heart Issues`= 'yes';

/* 3. Find the average hospitalization cost for each hospital tier and each city level*/

SELECT `Hospital tier`, AVG(charges) AS avg_cost FROM healthcare.hospitalisation_details GROUP BY `Hospital tier`;
SELECT `City tier`, AVG(charges) AS avg_cost FROM healthcare.hospitalisation_details GROUP BY `City tier`;

/* 4. Determine the number of people who have had major surgery with a history of cancer*/

SELECT COUNT(`Customer_Id`) FROM healthcare.medical_examinations WHERE 'Cancer history' = 'Yes' AND 'NumberOfMajorSurgeries'>0;

/* 5. Determine the number of tier-1 hospitals in each state */
SELECT `State ID`, COUNT(`Hospital tier`) FROM healthcare.hospitalisation_details WHERE `Hospital tier`='tier - 1' GROUP BY `State ID`;