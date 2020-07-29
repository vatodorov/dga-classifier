########################################################################
#
#   All right reserved (c)2020 - Valentin Todorov
#
#   Purpose: Read the downloaded data and prepare it for analysis
#
########################################################################

import pickle

# Logging


def read_data(analysis_file_loc):
    """
    Reads the DGA and non-DGA files for the analysis

    :param analysis_file_loc str: Location of the file with all the domains. In a pickle format
    :return data DataFrame: Analysis file in Pandas dataframe format
    """

    print('Starting to read the analysis file from {}, and build a sample for analysis...'.format(analysis_file_loc))
    with open(analysis_file_loc, 'rb') as f:
        data = pickle.load(f, encoding='bytes')

    print('Successfully read the file')

    return data


'/Users/valentint/Documents/GitRepos/dga-classifier/data/dga_sample.pkl'
'/Users/valentint/Documents/GitRepos/dga-classifier/data/non_dga_sample.pkl'

