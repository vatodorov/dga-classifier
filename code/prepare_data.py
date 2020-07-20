########################################################################
#
#   All right reserved (c)2020 - Valentin Todorov
#
#   Purpose: Read the downloaded data and prepare it for analysis
#
########################################################################


import pandas as pd
import zipfile
import os
import re

# Logging



def get_files(files_path):
    """
    Reads the file names from the provided path

    :param files_path str: Path to the files with domain names to be processed
    :return list: List of file names located at the provided path
    """

    # Sanitize the path
    # Removes the forward slash at the end, if it's provided in the input
    files_path = os.path.dirname(files_path)

    files = ['{}/{}'.format(files_path, x) for x in os.listdir(files_path)]

    return files


def read_archive(filename, file_extension):
    """
    Reads files from archives - ZIP
    If needed, add support for GZ and TAR in the future

    :param filename str: Filename of the archive
    :param file_extension str: Type of archive
    :return out_file: The extracted file
    """

    # For now assume that there is only one file in an archive
    if file_extension.lower() == 'zip':
        archive = zipfile.ZipFile(filename, 'r')
        out_file = archive.open(archive.namelist()[0])
    elif file_extension.lower() == 'gz':
        pass
    elif file_extension.lower() == 'tar':
        pass
    else:
        print('The archive format {} is not supported'.format(file_extension))

    return out_file


def get_data(filename, sep=None, skiprows=None, header=None, usecols=None, names=None):
    """
    Helper function that uses Pandas to create a data frame from the files

    :param filename str: Path to the file with the domains
    :param sep str: The type of separator in the files
    :param skiprows int: Number of rows to skip when reading the file
    :param header list: Row number(s) to use as the column names, and the start of the data
    :param usecols list: Index of the columns to read in
    :param names list: A list with column names assigned to the dataframe
    :return df_out Pandas DF: The Pandas dataframe that is returned
    """

    return (pd.read_csv(filename, sep=sep, skiprows=skiprows, header=header, na_values=['.'], usecols=usecols, names=names))


def read_data(filename, source):
    """
    Read in the downloaded files from:
        1) Alexa - alexa.csv.zip
        2) Majestic Million - mm.csv
        3) Cisco Umbrella - cisco.csv.zip
        4) DomCop - domcop.csv.zip
        5) Netlab 360 - netlab360.txt
        6) Bambenek DGA - bambenek_dga.txt
        7) Bambenek High Confidence DGA - bambenek_dga_hc.csv

    :param filename str: Path to the filename on the disk
    :param source str: Name of the source. This is extracted from the filename
    :return df_out Pandas DF: The Pandas data frame created from the raw data
    """

    data_processing_map = {
        'alexa': get_data(filename, skiprows=None, header=None, usecols=[0, 1], names=['domain_rank', 'domain']),
        'mm': get_data(filename, skiprows=1, header=None, usecols=[0, 2], names=['domain_rank', 'domain']),
        'cisco': get_data(filename, skiprows=None, header=None, usecols=[0, 1], names=['domain_rank', 'domain']),
        'domcop': get_data(filename, skiprows=1, header=None, usecols=[0, 1], names=['domain_rank', 'domain']),
        'bambenek_dga_hc': get_data(filename, skiprows=15, header=None, usecols=[0, 1, 2], names=['domain', 'malware', 'date']),
        'bambenek_dga': get_data(filename, sep=",", skiprows=15, header=None, usecols=[0, 1, 2], names=['domain', 'malware', 'date']),
        'netlab360': get_data(filename, sep="\t", skiprows=18, header=None, usecols=[0, 1, 2], names=['malware', 'domain', 'date'])
    }

    return (data_processing_map(source))


def create_target(row):
    """
    Creates the target variable
    (very inefficient, may need to think of doing it differently)

    :param row str: The row number name to use for creating the target variable
    :return int: Returns 1 if a domain is a DGA and 0 otherwise
    """

    if row['source'] == 'netlab360' or \
            row['source'] == 'bambenek_dga' or \
            row['source'] == 'bambenek_dga_hc' or \
            row['source'] == 'gen_dga':
        return 1
    else:
        return 0


def execute(files_path, sources):
    """
    Executes the complete pipeline to read in the data

    :param files_path str: Location of the files to read in
    :param sources list: List of sources to process
    :return:
    """

    # Reads the file names from the provided path
    files = get_files(files_path)

    # Read the domains from the files
    df = []
    for f in files:
        source = re.sub(r'\d+\-', '', os.path.basename(f)).split('.')[0]
        if source in sources:
            df.append(read_data(f, source))

    # Append the data frame to the main DF
    df = pd.concat(df)

    # Create the target variable
    df['dga_domain'] = df.apply(lambda x: create_target(x), axis=1)

    # Generate DGAs using the Python scripts from GitHub - Postpone for later (!!)

    # Dedup the dataframe by domain
    #   Are there dups by domain but different sources?


    # Store the final CSV with data
    output_data = df.to_csv('/Users/valentint/Documents/GitRepos/dga-classifier/data/project-data.csv', index=False)

    return


execute(
    files_path='/Users/valentint/Documents/GitRepos/dga-classifier/data/',
    sources=['alexa', 'cisco', 'domcop', 'mm', 'netlab360', 'bambenek_dga', 'bambenek_dga_hc'])


