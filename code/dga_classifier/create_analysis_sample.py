########################################################################
#
#   All right reserved (c)2020 - Valentin Todorov
#
#   Purpose: Read the downloaded data and prepare it for analysis
#
########################################################################

import pandas as pd
import pickle

# Logging


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


def combined_file_stats(analysis_file_loc, target):
    """
    Prints stats for the combined file

    :param analysis_file_loc str: Path to the combined data file with all domains
    :param target str: Name of the target variable in the file
    """

    data = read_data(analysis_file_loc)

    # This covers only the case when the resulting dataframe is larger than the required file size for analysis
    #   There can be other cases which are not currently covered
    cases_cnt = sum(data[target])
    rows_cnt = len(data[target])
    curr_inc_rate = round(cases_cnt / rows_cnt, 2)

    print('Combined file stats: \n' 
          '  -> Total number of records: {} \n'
          '  -> Number of cases: {} \n'
          '  -> Incidence rate: {}'.format(rows_cnt, cases_cnt, curr_inc_rate))


def build_sample(analysis_file_loc, out_file_loc, overwrite, target, max_file_size, incidence_rate, seed=7894, replace=False):
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

    # This covers only the case when the resulting dataframe is larger than the required file size for analysis
    #   There can be other cases which are not currently covered
    cases_cnt = sum(data[target])
    rows_cnt = len(data[target])
    curr_inc_rate = round(cases_cnt / rows_cnt, 2)

    print('Combined file stats: \n' 
          '  -> Total number of records: {} \n'
          '  -> Number of cases: {} \n'
          '  -> Incidence rate: {}'.format(rows_cnt, cases_cnt, curr_inc_rate))

    if rows_cnt > max_file_size:
        print('The source datafile {} is larger than the maximum requested of {}'.format(rows_cnt, max_file_size))
        new_cases = int(max_file_size * incidence_rate)

        if curr_inc_rate > incidence_rate:
            print('The incidence rate {} is higher than the requested rate {}'. format(curr_inc_rate, incidence_rate))
            df_dga = data[data[target] == 1].sample(n=new_cases, replace=replace, random_state=seed)
            df_non_dga = data[data[target] == 0].sample(n=(max_file_size - len(df_dga[target])), replace=replace, random_state=seed)

        elif curr_inc_rate < incidence_rate:
            print('The incidence rate {} is lower than the requested rate {}'.format(curr_inc_rate, incidence_rate))

            if cases_cnt > new_cases:
                print('The number of cases {} is higher than the requested count {}'.format(cases_cnt, new_cases))
                df_dga = data[data[target] == 1].sample(n=new_cases, replace=replace, random_state=seed)
                df_non_dga = data[data[target] == 0].sample(n=(max_file_size - len(df_dga[target])), replace=replace, random_state=seed)
            elif cases_cnt <= new_cases:
                print('The number of cases {} is less than or equal to the requested count {}'.format(cases_cnt, new_cases))
                df_dga = data[data[target] == 1]
                df_non_dga = data[data[target] == 0].sample(n=(max_file_size - len(df_dga[target])), replace=replace, random_state=seed)

        else:
            print('The incidence rate {} is equal to the requested rate {}'.format(curr_inc_rate, incidence_rate))

            df_dga = data[data[target] == 1]
            df_non_dga = data[data[target] == 0]

        dga_len = len(df_dga[target])
        non_dga_len = len(df_non_dga[target])
        print('The final count of cases and non-cases in the file is {} and {} - an incidence rate of {}.'.format(dga_len, non_dga_len, dga_len/(dga_len+non_dga_len)))

    analytical_sample = pd.concat([df_dga, df_non_dga], axis=0)

    # Store the final CSV with data
    if overwrite:
        analytical_sample.to_pickle('{}/{}'.format(out_file_loc, 'analytical_sample'))
        print('Saved the analysis samples to {}'.format(out_file_loc))


# Print stats about the current combined file with all domains
combined_file_stats(
    analysis_file_loc='/Users/valentint/Documents/GitRepos/dga-classifier/data/dga-combined-data.pkl',
    target='dga_domain')

# Execute the pipeline to build the sample
# This saves the DGAs and non-DGAs as separate pickle files
build_sample(
    analysis_file_loc='/Users/valentint/Documents/GitRepos/dga-classifier/data/dga-combined-data.pkl',
    out_file_loc='/Users/valentint/Documents/GitRepos/dga-classifier/data',
    overwrite=True,
    target='dga_domain',
    max_file_size=200000,
    incidence_rate=0.25)
