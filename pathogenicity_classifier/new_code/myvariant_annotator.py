import argparse
import myvariant
import itertools
import pandas as pd
import numpy as np
import os
import time
from multiprocessing import Process, Manager, cpu_count

def merge_files(output, file_locs, debug):
    print("\nmerge myvariant info annotations to "+output)
    files = os.listdir(file_locs)
    keep_columns = ['query', 'cadd.encode.h3k4me1', 'dbnsfp.fathmm-mkl.coding_rankscore', 
                            'dbnsfp.mutationassessor.rankscore', 'cadd.encode.h3k27ac',  'dbnsfp.eigen-pc.raw_coding',
                            'cadd.phast_cons.primate', 'dbnsfp.genocanyon.score', 'cadd.encode.exp']

    if os.path.isfile(output):
        print("deleting existing %s file" % output)
        str_rm = "rm -f "+output
        os.system(str_rm)

    for filename in files:
        batch_input = os.path.join(file_locs, filename)
        batch_data = pd.read_csv(batch_input, sep='\t', low_memory = False)
        for column in keep_columns:
            if(column not in batch_data.columns.tolist()):
                batch_data[column] = np.NaN
                print("WARNING : myvariant.info batch file "+batch_input+" does not contain the column "+column)
        batch_data = batch_data[keep_columns]
        if not os.path.isfile(output):
            batch_data.to_csv(output, header =True, index=None, sep='\t')
        else: # else it exists so append without writing the header
            batch_data.to_csv(output, mode = 'a',header=False, index=None, sep='\t')


    #remove batch files and directory if debugging is turned off
    if not debug:
        print("deleting " + file_locs)
        str_rm = "rm -rf " + file_locs
        os.system(str_rm)

# def merge_files(output, file_locs, debug):
#     print("\nmerge myvariant info annotations to "+output)
#     files = os.listdir(file_locs)
#     # keep_columns = ['query', 'cadd.encode.h3k4me1', 'dbnsfp.fathmm-mkl.coding_rankscore', 
#                     # 'dbnsfp.mutationassessor.rankscore', 'cadd.encode.h3k27ac',  'dbnsfp.eigen-pc.raw_coding',
#                     # 'cadd.phast_cons.primate', 'dbnsfp.genocanyon.score', 'cadd.encode.exp']

#     if os.path.isfile(output):
#         print("deleting existing %s file" % output)
#         str_rm = "rm -f "+output
#         os.system(str_rm)

#     cols = set()
#     batch_names = []
#     for filename in files:
#         batch_input = os.path.join(file_locs, filename)
#         batch_names.append(batch_input)

#         batch_data = pd.read_csv(batch_input, sep='\t', low_memory = False)

#         cols.update(batch_data.columns.tolist())


#     for batch in batch_names:

#         batch_data = pd.read_csv(batch, sep='\t', low_memory = False)

#         diff = cols - set(batch_data.columns.tolist())

#         for column in diff:
#             batch_data[column] = np.NaN
#             # print("WARNING : myvariant.info batch file "+batch_input+" does not contain the column "+column)

#         if not os.path.isfile(output):
#             batch_data.to_csv(output, header =True, index=None, sep='\t')
#         else: # else it exists so append without writing the header
#             batch_data.to_csv(output, mode = 'a',header=False, index=None, sep='\t')    
    
#     #remove batch files and directory if debugging is turned off
#     if not debug:
#         print("deleting " + file_locs)
#         str_rm = "rm -rf " + file_locs
#         os.system(str_rm)

def annotate_variants(input_lines, proc_num, directory):
    
    mv = myvariant.MyVariantInfo()
    
    batch = []
    while True:

        tup = input_lines.get()
        _, line = tup

        if line == None:
            if len(batch) == 0:
                return
            output = mv.getvariants(batch, fields="dbnsfp,clinvar,evs,cadd,gwassnps,cosmic,docm,snpedia,emv,grasp,civic,cgi", as_dataframe = True)

            new_file = directory + "process_" + proc_num + "_output.txt"

            with open(new_file, 'w+') as new_f:
                output.to_csv(new_f, sep='\t', encoding='utf-8')

            return


        elements = line.strip().split("\t")
        chrom, start, ref, alt, end, vtype = elements
        formatted_query = "chr" + chrom + ":g." +start
        
        if vtype=='SNP':
            formatted_query += ref + ">" + alt
        else:
            formatted_query += "_" + end + vtype.lower()
            if vtype == 'INS':
                formatted_query += alt

        # print(formatted_query + "\t" + line)
        batch.append(formatted_query)
   
def main():
    
    print("\nGet MyVariantInfo annotations")
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_maf', type=str, default='data/test_input.maf', help='The input maf file')
    parser.add_argument('--output_maf', type=str, default='data/annotated.maf', help='The output annotated maf file')
    parser.add_argument('--debug', type=bool, nargs='?', const=True, default=False, help="Don't delete auxiliary files")
    args = parser.parse_args()

    # get the input/output file path
    input_maf = args.input_maf
    output_maf = args.output_maf
    debug = args.debug
    
    # get the actual file name (in case the path to a directory was passed)
    input_file_name = input_maf.split('/',-1)[-1]

    # get the index of where this name begins
    index = input_maf.rfind(input_file_name)

    # make a temporary file in that directory with the phrase "stripped_" prepended to the file name
    new_file_name = input_maf[:index] + "stripped_" + input_file_name

    # make directory for batch files
    if not os.path.exists(input_maf[:index]+'mini'):
        os.makedirs(input_maf[:index]+'mini')

    # read the input file, retain only the necessary info for annotation, and then write the file out
    input_maf_data = pd.read_csv(input_maf, sep = '\t', low_memory = False)

    if 'Tumor_Seq_Allele2' in input_maf_data:
        input_maf_data.rename(columns={'Tumor_Seq_Allele2':'Alternate_Allele'}, inplace=True)

    input_maf_data = input_maf_data.loc[:, ['Chromosome', 'Start_Position', 'Reference_Allele', 'Alternate_Allele', 'End_Position', 'Variant_Type']]
    input_maf_data.drop_duplicates(inplace=True)
    input_maf_data.to_csv(new_file_name, sep = '\t', index = None)
    time.sleep(15)
    num_proc = cpu_count()

    manager = Manager()
    work = manager.Queue(num_proc)

    # start for workers    
    pool = []
    for i in range(num_proc):
        p = Process(target=annotate_variants, args=(work, str(i), input_maf[:index]+'mini/'))
        p.start()
        pool.append(p)

    # produce data
    with open(new_file_name) as f:
        f.readline()
        iters = itertools.chain(f, (None,)*num_proc)
        for i in enumerate(iters):
            work.put(i)

    for p in pool:
        p.join()

    # remove the temporary file if debug flag is false
    if not debug:
        print("\ndeleting %s" % new_file_name)
        os.remove(new_file_name)
    
    merge_files(output_maf, input_maf[:index]+'mini/', debug)


if __name__ == '__main__':
	main()	