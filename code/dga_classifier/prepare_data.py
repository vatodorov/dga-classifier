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
import sys

# Logging


# Environment variables
dga_generators_loc = '/Users/valentint/Documents/GitRepos/dga-classifier/code/dga_generators/domain_generation_algorithms'
analysis_file_loc = '/Users/valentint/Documents/GitRepos/dga-classifier/data/dga-combined-data.pkl'
domains_data_loc = '/Users/valentint/Documents/GitRepos/dga-classifier/data/'


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

    print('Successfully read the list of files. Found {} files with the following names {}'.format(len(files), ',\n'.join(files)))

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


def get_data(filename, params, source):
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

    print('Starting to build a dataframe from file {}'.format(filename))
    df = pd.read_csv(
        filename,
        sep=params['sep'],
        skiprows=params['skiprows'],
        header=params['header'],
        na_values=['.'],
        usecols=params['usecols'],
        names=params['names'],
        dtype=params['dtype']
    )

    df['source'] = source
    print('Successfully, created a dataframe for file {}'.format(filename))

    return df


def create_target(row):
    """
    Creates the target variable
    (very inefficient, may need to think of doing it differently)

    :param row str: The row number name to use for creating the target variable
    :return int: Returns 1 if a domain is a DGA and 0 otherwise
    """

    if row['source'] == 'alexa' or \
            row['source'] == 'mm' or \
            row['source'] == 'cisco' or \
            row['source'] == 'domcop':
        return 0
    else:
        return 1


def execute(files_path, analysis_file, sources, dga_generators_loc, target, incidence_rate, overwrite=False, dga_generators=[]):
    """
    Executes the complete pipeline to read in the data

    :param files_path str: Location of the files to read in
    :param sources list: List of sources to process
    :return:
    """

    # Reads the file names from the provided path
    files = get_files(files_path)

    # IMPORTANT:
    #   For now, I'm only reading the domain name, but there is more that can be collected.
    #       -> Alexa, Cisco, MM, Domcop: domain_rank (type Int64)
    #       -> Bambenek, Bambenek DGA HC, Netlab360: malware (type str), date (type str)
    map_params = {
        'alexa': {'sep': ',', 'skiprows': None, 'header': None, 'usecols': [1],
                  'names': ['domain'],
                  'dtype': {'domain': str}},
        'mm': {'sep': ',', 'skiprows': 1, 'header': None, 'usecols': [2],
               'names': ['domain'],
               'dtype': {'domain': str}},
        'cisco': {'sep': ',', 'skiprows': None, 'header': None, 'usecols': [1],
                  'names': ['domain'],
                  'dtype': {'domain': str}},
        'domcop': {'sep': ',', 'skiprows': 1, 'header': None, 'usecols': [1],
                   'names': ['domain'],
                   'dtype': {'domain': str}},
        'bambenek_dga_hc': {'sep': ',', 'skiprows': 15, 'header': None, 'usecols': [0],
                            'names': ['domain'],
                            'dtype': {'domain': str}},
        'bambenek_dga': {'sep': ',', 'skiprows': 15, 'header': None, 'usecols': [0],
                         'names': ['domain'],
                         'dtype': {'domain': str}},
        'netlab360': {'sep': '\t', 'skiprows': 18, 'header': None, 'usecols': [1],
                      'names': ['domain'],
                      'dtype': {'domain': str}}
    }

    # Read the domains from the files
    df = []
    for f in files:
        source = re.sub(r'\d+\-', '', os.path.basename(f)).split('.')[0]
        if source in sources:
            print('Processing data for source {} from file {}'.format(source, f))
            data = get_data(f, map_params[source], source)
            print('Successfully processed the data from source {}'.format(source))
            print(data.head(10))

            df.append(data)

    # Generate DGAs using the Python scripts from GitHub - Postpone for later (!!)
    for d in dga_generators:
        execpy = '{}/{}'.format(dga_generators_loc, d)

        # Import the DGA generator
        sys.path.insert(0, execpy)
        import dga

        # This outputs a list
        data = dga.main(malware_family=d)
        data['source'] = 'banerj'
        df.append(data)

    # Append the dataframe to the main DF
    df = pd.concat(df)

    # Dedup the dataframe by domain
    #   Are there dups by domain but different sources?
    #   For now just remove all dups - regardless if they may belong to DGA and non-DGA
    df.drop_duplicates(subset=['domain'], keep='first', inplace=True)

    # Create the target variable
    df[target] = df.apply(lambda x: create_target(x), axis=1)
    print('The final dataframe is:')
    print(df.head(10))
    print(df.tail(10))
    print(df.describe(include='all'))

    # Store the final CSV with data
    if overwrite:
        df.to_pickle(analysis_file)

# Run the data aggregation pipeline
# There are also benign domains from 'domcop' but the file is 10 million records and not worth it for now
execute(
    files_path=domains_data_loc,
    analysis_file=analysis_file_loc,
    sources=['alexa', 'mm', 'cisco', 'netlab360', 'bambenek_dga', 'bambenek_dga_hc'],
    dga_generators_loc=dga_generators_loc,
    target='dga_domain',
    incidence_rate=0.2,
    overwrite=True,
    dga_generators=[]
)


def create_sample(df, target, incidence_rate):

    # Calculate the current incidence rate


    # Re-calculate how to change the


    # Draw sample from the non-dga


    # Draw sample from the dga


    # Combine the non-dga with the dga

    return df




