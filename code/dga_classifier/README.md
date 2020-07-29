# DGA Classifier



## Code execution sequence

Data Preparation:
* download_data.sh - Downloads domains data from Alexa, Cisco Umbrella, Majestic Million, 
    DomCop, Bambenek DGA, Bambenek High Confidence DGA, Netlab360
* prepare_data.py - Combines the downloaded files and creates the base analytical file
* create_analysis_sample.py - Builds the analytical sample from the base file which will be used by the model

Build Models:
* model_lstm.py - Builds an LSTM model
* model_cnn.py - Builds a CNN model
* model_bigram.py - Builds a bigram model

Summarize model results:
* asdf
* asdf

## Something, something

