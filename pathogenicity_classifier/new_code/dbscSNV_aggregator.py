import pandas as pd
import os
import argparse

def main():
    
    print ("\nGet dbscSNV annotations. This will take a while")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dbscSNV_dir', type=str, help='The directory where the dbscSNV files reside.')
    args = parser.parse_args()

    # get the dbscSNV file directory path
    dbscSNV_data = args.dbscSNV_dir
    
    # get the dbscSNV file names (for each chromosome) and make an empty dataframe with the relevant column names
    dbscsnv_files = os.listdir(dbscSNV_data)
    column_names = ['chr', 'pos', 'ref', 'alt', 'ada_score', 'rf_score']
    dbscSNV = pd.DataFrame(columns = column_names)

    print('')
    # for each file given the correct filename prefix, read the file into memory, and append it to the empty dataframe
    for filename in dbscsnv_files:
        dbscsnv_file = os.path.join(dbscSNV_data, filename)
        if(filename.startswith("dbscSNV1.1.chr")):
            print('Loading and appending %s' % dbscsnv_file)
            dbscSNV_1 = pd.read_csv(dbscsnv_file, sep='\t', compression = 'gzip', low_memory = False)
            dbscSNV_1 = dbscSNV_1[column_names]
            dbscSNV = dbscSNV.append(dbscSNV_1, ignore_index=True)

    print('\nSaving data into ALL_dbscSNV1.1.gz')
    dbscSNV.to_csv(dbscSNV_data + "/ALL_dbscSNV1.1.gz", sep = '\t', index = None,compression='gzip')
    print('Finished saving data into ALL_dbscSNV1.1.gz\n')

if __name__ == '__main__':
	main()	