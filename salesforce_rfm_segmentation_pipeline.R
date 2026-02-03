#### SALESFORCE RFM SEGMENTATION PIPELINE ####
  
# Purpose:
# Automates an end‑to‑end RFM segmentation workflow for nonprofit fundraising data.

# The script:
# (1) Fetches supporter and transaction data from Salesforce via API
# (2) Performs RFM scoring and k‑means clustering to generate donor segments
# (3) Updates selected Salesforce fields via API to reflect new segment assignments

# Key Features:
# - Uses Salesforce Bulk 1.0 API endpoints (no proprietary objects or credentials included)
# - Implements reproducible RFM scoring logic suitable for nonprofit fundraising contexts
# - Applies k‑means clustering to derive data‑driven donor segments
# - Includes placeholder variables for all sensitive configuration values

# Output:
# Updated Salesforce records (via API) and local R objects containing RFM scores and cluster labels.

# Author: ktevon
# Date: February 2026

# Get required packages
library(salesforcer)
library(rjson)
library(dplyr)
library(lubridate)

#### Fetch credential ####

login <- fromJSON(file = "login.txt") # JSON dictionary containing "username", "password", and "token" 

#### Connect to Salesforce ####

user <- login$username
password <- login$password
token <- login$token

password.token <- paste(password, token, sep = "")
session <- sf_auth(user, password.token)

#### Prepare for RFM ####

# Fetch records

soql_opps <- "SELECT Id, AccountId, ID18_Contact__c, npe03__Recurring_Donation__c, Amount, CloseDate, Campaign_Type__c,
                Account.Name, Account.RecordtypeId, npsp__Primary_Contact__r.Name
                FROM Opportunity
                WHERE StageName = 'Closed Won'
                  AND RecordTypeId = '0127F000001AR1RQAW'
                  AND Adjusted__c = FALSE
                  AND Campaign_Type__c != 'Bequest'
                  AND (NOT Account.Name LIKE '%Anonymous%')
                  AND Account.Name != 'CCIA Staff'
                  AND (NOT ASPHPP__Contact__r.Name LIKE '%Anonymous%')
                  AND ASPHPP__Contact__r.npsp__Primary_Affiliation__r.Name != 'CCI Internal Cases'
                  AND ASPHPP__Contact__r.Bequest_Status__c != 'Confirmed'
                  AND ASPHPP__Contact__r.npsp__Deceased__c = FALSE"

opps <- sf_query(soql_opps, object_name = "Opportunity", api_type = "Bulk 1.0")

# Define segments
champions <- c(555, 554, 544, 545, 454, 455, 445)
loyal_customers <- c(543, 444, 435, 355, 354, 345, 344, 335)
potential_loyalist <- c(553, 551,552, 541, 542, 533, 532, 531, 452, 451, 442, 441, 431, 453, 433, 432, 423, 353, 352, 351, 342, 341, 333, 323)
recent_customers <- c(512, 511, 422, 421, 412, 411, 311)
promising <- c(525, 524, 523, 522, 521, 515, 514, 513, 425, 424, 413,414, 415, 315, 314, 313)
needing_attention <- c(535, 534, 443, 434, 343, 334, 325, 324)
about_to_sleep <- c(331, 321, 312, 221, 213)
at_risk <- c(255, 254, 245, 244, 253, 252, 243, 242, 235, 234, 225, 224, 153, 152, 145, 143, 142, 135, 134, 133, 125, 124)
cant_lose <- c(155, 154, 144, 214, 215, 115, 114, 113)
hibernating <- c(332, 322, 231, 241, 251, 233, 232, 223, 222, 132, 123, 122, 212, 211)
lost <- c(111, 112, 121, 131, 141, 151)

# Opportunities last four years
opps <- opps %>% 
  filter(CloseDate <= Sys.Date() &
           CloseDate > Sys.Date() - years(4) &
           !is.na(AccountId) & # In case some Account IDs are blank
           !is.na(ID18_Contact__c) & # In case some Contact IDs are blank
           !is.na(Amount)) # In case some Amounts are blank

# Separate Accounts and Contacts
opps_acct <- opps %>% 
  filter(Account.RecordTypeId == "XXX") # Replace with your RecordTypeId

opps_cont <- opps %>% 
  filter(Account.RecordTypeId == "XXX") # Replace with your RecordTypeId

# Identify RG opportunities
rg_opps_cont <- opps_cont %>% 
  filter(!is.na(npe03__Recurring_Donation__c))

# Identify RG contacts
rg_cont <- rg_opps_cont %>% 
  distinct(ID18_Contact__c)

# Identify non-RG opportunities
non_rg_opps_cont <- opps_cont %>% 
  filter(is.na(npe03__Recurring_Donation__c))

# Identify non-RG contacts
non_rg_cont <- non_rg_opps_cont %>% 
  distinct(ID18_Contact__c)

# Identify contacts who are both rg and non-rg
rg_and_non_rg_cont <- rg_cont %>% 
  semi_join(non_rg_cont, by = "ID18_Contact__c")

# Opportunities for contacts who are both rg and non-rg
rg_and_non_rg_opps_cont <- opps_cont %>% 
  filter(ID18_Contact__c %in% rg_and_non_rg_cont$ID18_Contact__c)

# Opportunities for contacts who are rg only
rg_only_opps_cont <- rg_opps_cont %>% 
  anti_join(rg_and_non_rg_opps_cont, by = "ID18_Contact__c")

# Opportunities for contacts who are non-rg only
non_rg_only_opps_cont <- non_rg_opps_cont %>% 
  anti_join(rg_and_non_rg_opps_cont, by = "ID18_Contact__c")

#### Run RFM ####

run_rfm <- function(data, id, group, file){
  
  id_quoted <- as.name(id)
  id_quoted <- enquo(id_quoted)
  
  df <- data %>% 
    group_by(!!id_quoted) %>% 
    summarise(`Last Gift Date` = max(CloseDate),
              Frequency = n(),
              Monetary = sum(Amount)) %>% 
    ungroup() %>% 
    mutate(Recency = as.numeric(
      difftime(Sys.Date(), `Last Gift Date`, units = c("days")))) %>% 
    filter(Monetary > 0) # Remove zero and negative amounts here in case they are still in the data frame
  
  scaled_df <- as.data.frame(df) # Convert tibble to data frame
  
  rownames(scaled_df) <- scaled_df$id # Set row names
  
  scaled_df$id <- NULL # Remove id
  
  scaled_df <- scaled_df %>% 
    select(-`Last Gift Date`) %>% 
    mutate_at(c("Recency", "Frequency", "Monetary"), ~(scale(.) %>% as.vector)) # Scale (though not necessarily required)
  
  set.seed(123)
  
  k5_r <- scaled_df %>% # K-means
    select(Recency) %>% 
    kmeans(centers = 5, nstart = 25)
  
  k5_f <- scaled_df %>% # K-means
    select(Frequency) %>% 
    kmeans(centers = 5, nstart = 25)
  
  k5_m <- scaled_df %>% # K-means
    select(Monetary) %>% 
    kmeans(centers = 5, nstart = 25)
  
  df %>% 
    mutate(r_score_temp = k5_r$cluster,
           f_score_temp = k5_f$cluster,
           m_score_temp = k5_m$cluster) %>% 
    group_by(r_score_temp) %>% 
    mutate(r_mean = mean(Recency)) %>%
    ungroup() %>% 
    group_by(f_score_temp) %>% 
    mutate(f_mean = mean(Frequency)) %>%
    ungroup() %>% 
    group_by(m_score_temp) %>% 
    mutate(m_mean = mean(Monetary)) %>% 
    ungroup() %>% 
    mutate(`R Score` = dense_rank(-r_mean), # Windowed rank function to reorder clusters.
           # The minus sign is equivalent to "ties.method = 'min'", which assigns every tied element to the lowest rank.
           # Presumably the script works without the minus sign.
           `F Score` = dense_rank(desc(-f_mean)),
           `M Score` = dense_rank(desc(-m_mean)),
           `RFM Score` = 100 * `R Score` + 10 * `F Score` + `M Score`) %>% 
    select(-`Last Gift Date`, -r_score_temp, -f_score_temp, -m_score_temp) %>% 
    mutate(Segment =
             case_when(
               `RFM Score` %in% champions ~ paste0("Champions", " / ", group),
               `RFM Score` %in% loyal_customers ~ paste0("Loyal Customers", " / ", group),
               `RFM Score` %in% potential_loyalist ~ paste0("Potential Loyalist", " / ", group),
               `RFM Score` %in% recent_customers ~ paste0("Recent Customers", " / ", group),
               `RFM Score` %in% promising ~ paste0("Promising", " / ", group),
               `RFM Score` %in% needing_attention ~ paste0("Customer Needing Attention", " / ", group),
               `RFM Score` %in% about_to_sleep ~ paste0("About to Sleep", " / ", group),
               `RFM Score` %in% at_risk ~ paste0("At Risk", " / ", group),
               `RFM Score` %in% cant_lose ~ paste0("Can't Lose Them", " / ", group),
               `RFM Score` %in% hibernating ~ paste0("Hibernating", " / ", group),
               `RFM Score` %in% lost ~ paste0("Lost", " / ", group)
             ))
}

rfm_org <- run_rfm(opps_acct, "AccountId", "Organisation", "Organisations")
rfm_rg_only <- run_rfm(rg_only_opps_cont, "ID18_Contact__c", "RG Only", "RG Only")
rfm_non_rg_only <- run_rfm(non_rg_only_opps_cont, "ID18_Contact__c", "Non-RG Only", "Non-RG Only")
rfm_rg_and_non_rg <- run_rfm(rg_and_non_rg_opps_cont, "ID18_Contact__c", "RG and Non-RG", "RG and Non-RG")

#### Import to Salesforce ####

# Append Commitment Scores
score_org <- rfm_org %>% 
  mutate(`Commitment Score` = case_when(
    Segment == "Champions / Organisation"	~ 1050,
    Segment == "Loyal Customers / Organisation"	~ 950,
    Segment == "Potential Loyalist / Organisation"	~ 850,
    Segment == "Recent Customers / Organisation"	~ 750,
    Segment == "Promising / Organisation"	~ 650,
    Segment == "Customer Needing Attention / Organisation"	~ 550,
    Segment == "About to Sleep / Organisation"	~ 450,
    Segment == "At Risk / Organisation"	~ 350,
    Segment == "Can't Lose Them / Organisation"	~ 250,
    Segment == "Hibernating / Organisation"	~ 150,
    Segment == "Lost / Organisation"	~ 50,
    TRUE ~ NA_real_))

score_rg_only <- rfm_rg_only %>% 
  mutate(`Commitment Score` = case_when(
    Segment == "Champions / RG Only" ~ 989,
    Segment == "Loyal Customers / RG Only" ~ 889,
    Segment == "Potential Loyalist / RG Only" ~ 789,
    Segment == "Recent Customers / RG Only" ~ 689,
    Segment == "Promising / RG Only" ~ 589,
    Segment == "Customer Needing Attention / RG Only" ~ 489,
    Segment == "About to Sleep / RG Only" ~ 389,
    Segment == "Can't Lose Them / RG Only" ~ 149,
    Segment == "At Risk / RG Only" ~ 249,
    Segment == "Hibernating / RG Only" ~ 89,
    Segment == "Lost / RG Only" ~	19,
    TRUE ~ NA_real_))

score_non_rg_only <- rfm_non_rg_only %>% 
  mutate(`Commitment Score` = case_when(
    Segment == "Champions / Non-RG Only"	~ 899,
    Segment == "Loyal Customers / Non-RG Only"	~ 799,
    Segment == "Potential Loyalist / Non-RG Only" ~	699,
    Segment == "Recent Customers / Non-RG Only" ~	599,
    Segment == "Promising / Non-RG Only"	~ 499,
    Segment == "Customer Needing Attention / Non-RG Only"	~ 399,
    Segment == "About to Sleep / Non-RG Only"	~ 299,
    Segment == "At Risk / Non-RG Only"	~ 199,
    Segment == "Can't Lose Them / Non-RG Only"	~ 99,
    Segment == "Hibernating / Non-RG Only"	~ 49,
    Segment == "Lost / Non-RG Only"	~ 0.5,
    TRUE ~ NA_real_))

score_rg_and_non_rg <- rfm_rg_and_non_rg %>% 
  mutate(`Commitment Score` = case_when(
    Segment == "Champions / RG and Non-RG"	~ 949,
    Segment == "Loyal Customers / RG and Non-RG"	~ 849,
    Segment == "Potential Loyalist / RG and Non-RG"	~ 749,
    Segment == "Recent Customers / RG and Non-RG"	~ 649,
    Segment == "Promising / RG and Non-RG"	~ 549,
    Segment == "Customer Needing Attention / RG and Non-RG"	~ 419,
    Segment == "About to Sleep / RG and Non-RG"	~ 319,
    Segment == "At Risk / RG and Non-RG"	~ 219,
    Segment == "Can't Lose Them / RG and Non-RG"	~ 109,
    Segment == "Hibernating / RG and Non-RG"	~ 69,
    Segment == "Lost / RG and Non-RG" ~ 	9,
    TRUE ~ NA_real_))

# Blank out existing Commitment Scores: Account
soql_acct <- "SELECT Id FROM Account WHERE Commitment_Score__c != NULL"

queried_acct <- sf_query(soql_acct, object_name = "Account", api_type = "Bulk 1.0")

updated_acct <- tibble(
  Id = queried_acct$Id,
  Commitment_Score__c = NA_real_
)

sf_update(updated_acct, object_name = "Account", api_type = "Bulk 1.0")

# Blank out existing Commitment Scores: Contact
soql_cont <- "SELECT Id FROM Contact WHERE Commitment_Score__c != NULL"

queried_cont <- sf_query(soql_cont, object_name = "Contact", api_type = "Bulk 1.0")

updated_cont <- tibble(
  Id = queried_cont$Id,
  Commitment_Score__c = NA_real_
)

sf_update(updated_cont, object_name = "Contact", api_type = "Bulk 1.0")

# Update the Commitment Score field: Organisation
updated_org <- score_org %>% 
  select(Id = AccountId,
         Commitment_Score__c = `Commitment Score`)

sf_update(updated_org, object_name = "Account", api_type = "Bulk 1.0")

# Update the Commitment Score field: RG Only
updated_rg_only <- score_rg_only %>% 
  select(Id = ID18_Contact__c,
         Commitment_Score__c = `Commitment Score`)

sf_update(updated_rg_only, object_name = "Contact", api_type = "Bulk 1.0")
  

# Update the Commitment Score field: Non-RG Only
updated_non_rg_only <- score_non_rg_only %>% 
  select(Id = ID18_Contact__c,
         Commitment_Score__c = `Commitment Score`)

sf_update(updated_non_rg_only, object_name = "Contact", api_type = "Bulk 1.0")


# Update the Commitment Score field: Non-RG Only
updated_rg_and_non_rg <- score_rg_and_non_rg %>% 
  select(Id = ID18_Contact__c,
         Commitment_Score__c = `Commitment Score`)

sf_update(updated_rg_and_non_rg, object_name = "Contact", api_type = "Bulk 1.0")