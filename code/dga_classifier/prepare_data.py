########################################################################
#
#   All right reserved (c)2020 - Valentin Todorov
#
#   Purpose: Read the downloaded data and prepare it for analysis
#
########################################################################

import os
import pandas as pd
import pickle
import re
import sys
import zipfile

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
        return 'non_dga'
    else:
        return 'dga'


def execute(files_path, analysis_file, sources, dga_generators_loc, target, overwrite=False, dga_generators=[]):
    """
    Executes the complete pipeline to read in the data

    :param files_path str: Location of the files to read in
    :param sources list: List of sources to process
    :return:
    """

    # Reads the file names from the provided path
    files = get_files('{}/'.format(files_path))

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


def read_data(analysis_file_loc):
    """
    Reads the file with all the combined domains
    This file is created using the prepare_data.py script

    :param analysis_file_loc str: Location of the file with all the domains. In a pickle format
    :return data DataFrame: Analysis file in Pandas dataframe format
    """

    print('Starting to read the combined data file from {}, and build a sample for analysis...'.format(analysis_file_loc))
    with open(analysis_file_loc, 'rb') as f:
        data = pickle.load(f, encoding='bytes')

    print('Successfully read the combined data file')

    return data


def combined_file_stats(analysis_file_loc, target, target_dga):
    """
    Prints stats for the combined file

    :param analysis_file_loc str: Path to the combined data file with all domains
    :param target str: Name of the target variable in the file
    """

    data = read_data(analysis_file_loc)

    # This covers only the case when the resulting dataframe is larger than the required file size for analysis
    #   There can be other cases which are not currently covered
    cases_cnt = len(data[data[target] == target_dga])
    rows_cnt = len(data[target])
    curr_inc_rate = round(cases_cnt / rows_cnt, 2)

    print('Combined file stats: \n' 
          '  -> Total number of records: {} \n'
          '  -> Number of cases: {} \n'
          '  -> Incidence rate: {}'.format(rows_cnt, cases_cnt, curr_inc_rate))


def build_sample(analysis_file_loc, out_file_loc, out_file, overwrite, target, target_dga, max_file_size, incidence_rate, seed=7894, replace=False):
    """
    Builds the analytical sample. Implements methods for resampling of both cases and non-cases.
    Gives the ability to set a custom incidence rate and the maximum records in the sample

    :param analysis_file_loc str: Path to the combined data file with all domains
    :param out_file_loc str: Path to save the DGA and non-DGA files for the models
    :param target str: Name of the target variable in the file
    :param max_file_size int: The maximum file size for the analysis
    :param incidence_rate float: The desired incidence rate in the analysis file
    :param seed int: The seed for the sampling
    :param replace bool: Sample with or without replacement. Default is without replacement (False)
    :return dga_sample non_dga_sample pickle:
    """

    data = read_data(analysis_file_loc)
    _target = ''.join([x for x in data[target].value_counts().index if x != target_dga])

    # This covers only the case when the resulting dataframe is larger than the required file size for analysis
    #   There can be other cases which are not currently covered
    cases_cnt = len(data[data[target] == target_dga])
    rows_cnt = len(data[target])
    curr_inc_rate = round(cases_cnt / rows_cnt, 2)

    print('Combined file stats: \n' 
          '  -> Total number of records: {} \n'
          '  -> Number of cases: {} \n'
          '  -> Incidence rate: {}'.format(rows_cnt, cases_cnt, curr_inc_rate))



    # TODO: Data sanitization
    #   - Drop any records longer than 253. The maximum domain length is 253
    #   - Drop records that start with special chars
    #   - Split the domain name in labels by the dots, and drop records with labels longer than 63 chars




    # Resample the file
    if rows_cnt > max_file_size:
        print('The source datafile {} is larger than the maximum requested of {}'.format(rows_cnt, max_file_size))
        new_cases = int(max_file_size * incidence_rate)

        if curr_inc_rate > incidence_rate:
            print('The incidence rate {} is higher than the requested rate {}'. format(curr_inc_rate, incidence_rate))
            df_dga = data[data[target] == target_dga].sample(n=new_cases, replace=replace, random_state=seed)
            df_non_dga = data[data[target] == _target].sample(n=(max_file_size - len(df_dga[target])), replace=replace, random_state=seed)

        elif curr_inc_rate < incidence_rate:
            print('The incidence rate {} is lower than the requested rate {}'.format(curr_inc_rate, incidence_rate))

            if cases_cnt > new_cases:
                print('The number of cases {} is higher than the requested count {}'.format(cases_cnt, new_cases))
                df_dga = data[data[target] == target_dga].sample(n=new_cases, replace=replace, random_state=seed)
                df_non_dga = data[data[target] == _target].sample(n=(max_file_size - len(df_dga[target])), replace=replace, random_state=seed)
            elif cases_cnt <= new_cases:
                print('The number of cases {} is less than or equal to the requested count {}'.format(cases_cnt, new_cases))
                df_dga = data[data[target] == target_dga]
                df_non_dga = data[data[target] == _target].sample(n=(max_file_size - len(df_dga[target])), replace=replace, random_state=seed)

        else:
            print('The incidence rate {} is equal to the requested rate {}'.format(curr_inc_rate, incidence_rate))

            df_dga = data[data[target] == target_dga]
            df_non_dga = data[data[target] == _target]

        dga_len = len(df_dga[target])
        non_dga_len = len(df_non_dga[target])
        print('The final count of cases and non-cases in the file is {} and {} - an incidence rate of {}.'.format(dga_len, non_dga_len, dga_len/(dga_len+non_dga_len)))

    analytical_sample = pd.concat([df_dga, df_non_dga], axis=0)

    # Shuffle the ordering of the dataframe
    analytical_sample.sample(frac=1).reset_index(drop=True)

    # Store the final CSV with data
    if overwrite:
        analytical_sample.to_pickle('{}/{}'.format(out_file_loc, out_file))
        print('Saved the analysis sample to {}/{}'.format(out_file_loc, out_file))


# ============================================================================================================= #

# Environment variables
dga_generators_loc = '/Users/valentint/Documents/GitRepos/dga-classifier/code/dga_generators/domain_generation_algorithms'
analysis_file = 'dga-combined-data.pkl'
domains_data_loc = '/Users/valentint/Documents/GitRepos/dga-classifier/data'
out_file = 'analytical_sample.pkl'
target = 'dga_domain'

# Run the data aggregation pipeline
# There are also benign domains from 'domcop' but the file is 10 million records and not worth it for now
execute(
    files_path=domains_data_loc,
    analysis_file='{}/{}'.format(domains_data_loc, analysis_file),
    sources=['alexa', 'mm', 'cisco', 'netlab360', 'bambenek_dga', 'bambenek_dga_hc'],
    dga_generators_loc=dga_generators_loc,
    target=target,
    overwrite=True,
    dga_generators=[]
)


# Print stats about the current combined file with all domains
combined_file_stats(
    analysis_file_loc='{}/{}'.format(domains_data_loc, analysis_file),
    target=target,
    target_dga='dga')


# Execute the pipeline to build the sample
# This saves the DGAs and non-DGAs as separate pickle files
build_sample(
    analysis_file_loc='{}/{}'.format(domains_data_loc, analysis_file),
    out_file_loc=domains_data_loc,
    out_file=out_file,
    overwrite=True,
    target=target,
    target_dga='dga',
    max_file_size=300000,
    incidence_rate=0.25)



