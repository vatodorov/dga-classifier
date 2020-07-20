########################################################################
#
#   All right reserved (c)2020 - Valentin Todorov
#
#   Purpose: Read the downloaded data and prepare it for analysis
#
########################################################################


import pandas as pd









# Read in the downloaded files from:
#   1) Alexa
#   2) Majestic Million
#   3) Cisco Umbrella
#   4) DomCop
#   5) Netlab 360
#   6) Bambenek DGA
#   7) Bambenek High Confidence DGA



# Alexa


# Majestic Million


# Cisco Umbrella


# DomCop


# Netlab 360


# Bambenek DGA


# Bambenek High Confidence DGA





# Merge all tables into a single dataframe
df =


# Dedup the dataframe by domain
#   Are there dups by domain but different sources?


# Generate DGAs using the Python scripts from GitHub - maybe postpone for later



# Create the target variable
def create_target(row):
    if row['source'] == 'netlab360' or \
            row['source'] == 'bambenek-dga' or \
            row['source'] == 'bambenek-dga-hc' or \
            row['source'] == 'gen-dga':
        return 1
    else:
        return 0


df['dga_domain'] = df.apply(lambda x: create_target(x), axis=1)


# Store the final CSV with data
output_data = df.to_csv('/Users/valentint/Documents/GitRepos/GitHub/ga-classifier/data/project-data.csv', index=False)


