#!/bin/bash
########################################################################
#
#   All right reserved (c)2020 - Valentin Todorov
#
#   Purpose: Download domain names from multiple sources
#
# #######
# Sources of data:
#
#  Non-DGA:
#     1) Alexa
#         http://s3.amazonaws.com/alexa-static/top-1m.csv.zip
#
#    2) Majestic Million
#        http://downloads.majestic.com/majestic_million.csv
#
#    3) Cisco Umbrella
#        http://s3-us-west-1.amazonaws.com/umbrella-static/top-1m.csv.zip
#
#    4) DomCop
#        https://www.domcop.com/files/top/top10milliondomains.csv.zip
#
#  DGAs
#    1) Netlab 360
#        http://data.netlab.360.com/feeds/dga/dga.txt
#
#    2) Bambenek
#        https://osint.bambenekconsulting.com/feeds/dga-feed.txt
#        https://osint.bambenekconsulting.com/feeds/dga-feed-high.csv
#
########################################################################

# Download the DGA and non-DGA domains


# Environment variables
data_loc="/Users/valentint/Documents/GitRepos/GitHub/ga-classifier/data/"
log_file="/Users/valentint/Documents/GitRepos/GitHub/ga-classifier/logs/log.txt"
time_now=$(date +%Y-%m-%dT%H:%M:%S)
ddate=$(date +%Y-%m-%d)


# List the data sources
for i in "alexa http://s3.amazonaws.com/alexa-static/top-1m.csv.zip" \
"mm http://downloads.majestic.com/majestic_million.csv" \
"domcop https://www.domcop.com/files/top/top10milliondomains.csv.zip" \
"cisco http://s3-us-west-1.amazonaws.com/umbrella-static/top-1m.csv.zip" \
"net360 http://data.netlab.360.com/feeds/dga/dga.txt" \
"bambenek-dga https://osint.bambenekconsulting.com/feeds/dga-feed.txt" \
"bambenek-dga-hc https://osint.bambenekconsulting.com/feeds/dga-feed-high.csv"


# Download the data from each source
do
    {
        wget -O ${data_loc}${ddate}-${i} -nv --no-check-certificate
        echo "${time_now}: [+] Successfully downloaded the data for ${ddate}-${i}" >> ${log_file}

    } || {
        echo "${time_now}: [+] Unable to download the data for ${ddate}-${i}" >> ${log_file}
    }

done
