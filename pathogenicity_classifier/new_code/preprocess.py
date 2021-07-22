import pandas as pd
import numpy as np
import os
import sys
import argparse
import sys

ONCOKB_API_TOKEN = "a2c2ea05-c8c4-4dce-ad5a-0be1cc99b24d"

### AUXILIARY FUNCTIONS ###

def debug_print(message, debug_flag):
    if debug_flag:
        print(message)

def check_path(path, isDir):
    if isDir:
        if not os.path.isdir(path):
            sys.exit('Error: %s is not a directory' % path)
    else:
        if not os.path.exists(path):
            sys.exit('Error: the file %s does not exist in the annotation_files directory.' % path)

###########################

def get_gene_list(input_file):

    # dictionary of updated gene names
    gene_new_names = {'MLL': 'KMT2A', "MLL3": "KMT2C", "MLL2": "KMT2D", "MLL4": "KMT2B"}

    # read in the input file and assign the columns names
    gene_list_cv = pd.read_csv(input_file, sep='\t', header=None, low_memory = False)
    gene_list_cv.columns = ['region', 'cytoband', 'gene_details']

    #get the gene names from the correct column and replace it if it needs updating
    gene_list_cv['gene_name'] = gene_list_cv['gene_details'].str.split(":").str.get(0)
    gene_list_cv = gene_list_cv.replace({'gene_name': gene_new_names})

    # drop any rows that have missing values and return 1 of every value in a list
    gene_list_cv = gene_list_cv.dropna(axis=0)
    return gene_list_cv['gene_name'].unique().tolist()

def setup_globals():
    
    global input_maf, annotated_maf, scripts_dir, oncokb_scripts, gl_input, gf_map_input, dbscSNV_data, cv5, cv6, myVariant_merged, debug

    #create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_maf', type=str, help='The input maf file')
    parser.add_argument('--annotated_maf', type=str, default=None, help='The resultant annotated maf file')
    parser.add_argument('--scripts_dir', type=str, default=os.path.join(os.getcwd()), help='scripts dir')
    parser.add_argument('--debug', action='store_true', help='Show debugging statements. If no debugging desired, leave off this argument.')

    #parse the command line arguments and store them
    args = parser.parse_args()
    input_maf = args.input_maf
    annotated_maf = args.annotated_maf
    scripts_dir = args.scripts_dir
    debug = args.debug

    # this directory gets created at location where input file resides
    full_input_path = os.path.abspath(input_maf)
    working_dir = os.path.dirname(full_input_path)
    print("\nAll intermediate files will get created in this directory: "+working_dir)

    # if no argument for annotated_maf, place it in the same directory as the input file
    if annotated_maf is None:
        annotated_maf = working_dir + '/annotated.maf'
        print("The annotated file will be located in this directory: " + annotated_maf)

    # this is the path of the file that will be annotated using MyVariantInfo
    myVariant_merged = os.path.join(working_dir, "myVariantInfo_annotated.txt")

    # path where all the scripts (for oncokb annotation) and files (dbscSNV and MSK-IMPACT) exist
    check_path(scripts_dir, isDir = True)
    annotation_files = os.path.join(scripts_dir, "annotation_files")
    check_path(annotation_files, isDir = True)

    #path to various files/scripts used for annotation
    oncokb_scripts = os.path.join(scripts_dir, "oncokb-annotator")
    check_path(oncokb_scripts, isDir = True)

    gl_input = os.path.join(annotation_files, "gene_level_annotation.txt")
    check_path(gl_input, isDir = False)

    gf_map_input = os.path.join(annotation_files, "gene_function_other_genes.txt")
    check_path(gf_map_input, isDir = False)

    dbscSNV_data = os.path.join(annotation_files, "dbscSNV")
    check_path(dbscSNV_data, isDir = True)

    #path to annotation files from MSK-Impact
    gene_list_cv5_file = os.path.join(annotation_files, "IMPACT_cv5.gene_intervals.list.annotated")
    check_path(gene_list_cv5_file, isDir = False)

    gene_list_cv3_file = os.path.join(annotation_files, "IMPACT_cv3.gene_intervals.list.annotated")
    check_path(gene_list_cv3_file, isDir = False)

    gene_list_cv6_file = os.path.join(annotation_files, "IMPACT_cv6.gene_intervals.list.annotated")
    check_path(gene_list_cv6_file, isDir = False)

    # get a list of genes from the annotation files from MSK IMPACT
    cv5_gene_list = get_gene_list(gene_list_cv5_file)
    cv3_gene_list = get_gene_list(gene_list_cv3_file)
    cv6_gene_list = get_gene_list(gene_list_cv6_file)

    # get list of genes that are only in CV5 and not in CV3
    cv5 = set(cv5_gene_list).difference(set(cv3_gene_list))
    cv6 = set(cv6_gene_list).difference(set(cv5_gene_list))

def get_oncokb_annotations(inputmaf, input_df):

    print("\nGet OncoKB annotations")

    #create input and output file paths
    oncokb_input_file = inputmaf.replace(".maf","_oncokb_input.maf")
    oncokb_output_file = inputmaf.replace(".maf","_oncokb_output.maf")

    # create a new file with only the necessary columns for oncoKB annotation
    input_df_subset = input_df[['Hugo_Symbol','Variant_Classification', 'Protein_position', 'HGVSp_Short']]
    input_df_subset = input_df_subset.drop_duplicates()
    input_df_subset['Normal_Sample'] = "SAMPLE"
    input_df_subset.to_csv(oncokb_input_file, sep = '\t', index = None)

    # call the oncoKB annotation script 
    oncokb_cmd = "python3 MafAnnotator.py -a -i "+oncokb_input_file+" -o "+oncokb_output_file+" -b "+ONCOKB_API_TOKEN

    debug_print(oncokb_cmd, debug)

    # oncoKB needs to be run from it's own directory so change to that dir and then back
    
    # save current working directory
    wd = os.getcwd()

    # switch to directory of oncoKB since it needs to be run from its own directory
    os.chdir(oncokb_scripts)

    # run the command
    os.system(oncokb_cmd)

    # switch back to previous working directory
    os.chdir(wd)

    return oncokb_input_file, oncokb_output_file

def get_dbscCNV_annotations():

    # make the aggregate file if it does not exist
    if not os.path.exists(dbscSNV_data + "/ALL_dbscSNV1.1.gz"):
        # run the dbscSNV annotator 
        str_myvariant = "python3 dbscSNV_aggregator.py --dbscSNV_dir " + dbscSNV_data
        debug_print("\nRan the following command: " + str_myvariant, debug)
        os.system(str_myvariant)
    else:
        print("\nfile: " + dbscSNV_data + "/ALL_dbscSNV1.1.gz" + " exists already.")

    print("Loading dbscSNV file.")
    dbscSNV = pd.read_csv(dbscSNV_data + "/ALL_dbscSNV1.1.gz", sep='\t', compression = 'gzip', low_memory = False)   
    debug_print("The new dataframe has the shape " + str(dbscSNV.shape), debug)
    debug_print("Head of the dbscSNV dataframe\n" + str(dbscSNV.head()), debug)

    return dbscSNV

def join_annotations(oncokb_input, oncokb_output, input, dbscSNV):

    input_data = input.copy()

    print("\nMerge the MyVariantInfo, OncoKB, and dbscSNV files")

    # load in both annotated files
    oncoKB_maf = pd.read_csv(oncokb_output, sep = '\t', low_memory = False)
    debug_print(oncokb_output + " shape is " +str(oncoKB_maf.shape), debug)

    variantinfo_maf = pd.read_csv(myVariant_merged, sep = '\t', low_memory = False)
    debug_print(myVariant_merged + " shape is " + str(variantinfo_maf.shape), debug)

    if not debug:
        # delete myvariant output file
        print("\ndeleting %s" % myVariant_merged)
        os.remove(myVariant_merged)

        # delete oncokb input file
        print("\ndeleting %s" % oncokb_input)
        os.remove(oncokb_input)

        #delete oncokb output file
        print("\ndeleting %s" % oncokb_output)
        os.remove(oncokb_output)

    # merge with OncoKB
    # make the mutation column in both annotation files 
    
    oncoKB_maf['mutation']= oncoKB_maf['Hugo_Symbol']+":"+oncoKB_maf['HGVSp_Short']+":"+oncoKB_maf['Protein_position'].astype(str)
    input_data['mutation']= input_data['Hugo_Symbol']+":"+input_data['HGVSp_Short']+":"+input_data['Protein_position'].astype(str)

    # isolate columns that provide useful information and remove any null values for the mutation column
    oncoKB_maf = oncoKB_maf.rename(columns = {'ONCOGENIC':'oncogenic', 'IS-A-HOTSPOT': 'is-a-hotspot', 'IS-A-3D-HOTSPOT': 'is-a-3d-hotspot'})
    
    oncoKB_maf_subset = oncoKB_maf[['mutation', 'oncogenic', 'is-a-hotspot', 'is-a-3d-hotspot']]
    oncoKB_maf_subset = oncoKB_maf_subset[~(pd.isnull(oncoKB_maf_subset['mutation']))]

    # merge the input dataframe and oncoKB dataset on the mutation column, adding oncoKB_maf_subset columns to input_data
    input_data = pd.merge(input_data, oncoKB_maf_subset, on = 'mutation', how = 'left')

    #at each position in the following columns, where the corresponding mutation value is null, set to 0, otherwise keep it what it was
    input_data['oncogenic'] = np.where(pd.isnull(input_data['mutation']), 0, input_data['oncogenic'])
    input_data['is-a-hotspot'] = np.where(pd.isnull(input_data['mutation']), 0, input_data['is-a-hotspot'])
    input_data['is-a-3d-hotspot'] = np.where(pd.isnull(input_data['mutation']), 0, input_data['is-a-3d-hotspot'])

    # drop duplicate values and change chromosome column to string type
    input_data = input_data.drop_duplicates()
    input_data['Chromosome'] = input_data['Chromosome'].astype(str)
    
    # merge with myvariant.info
    # at each position in the input data where we have a SNP, DEL, or INS, create an entry using the myvariantinfo format 
    input_data['query'] = np.where(input_data['Variant_Type']=='SNP',
                        "chr"+input_data['Chromosome']+":g."+input_data['Start_Position'].astype(str)+input_data['Reference_Allele']+">"+input_data['Alternate_Allele'],
                            np.where(input_data['Variant_Type']=='DEL',
                                "chr"+input_data['Chromosome']+":g."+input_data['Start_Position'].astype(str)+"_"+input_data['End_Position'].astype(str)+"del", #DEL,
                                "chr"+input_data['Chromosome']+":g."+input_data['Start_Position'].astype(str)+"_"+input_data['End_Position'].astype(str)+"ins"+input_data['Alternate_Allele'] #INS
                                ))

    # merge the input dataframe and myvariantinfo dataset on the query column, adding variantinfo_maf columns to input_data
    input_data = pd.merge(input_data, variantinfo_maf, on=['query'], how='left')
    debug_print("\nThe shape of the myvariant + oncoKB merged file is " + str(input_data.shape), debug)
    
    # merge with dbscSNV
    # change the names of the columns to match the input_data dataframe and cast them to the correct types 
    dbscSNV.columns = ['Chromosome','Start_Position', 'Reference_Allele',  'Alternate_Allele', 'ada_score', 'rf_score']
    dbscSNV['Chromosome'] = dbscSNV['Chromosome'].astype(str)
    dbscSNV['Start_Position'] = dbscSNV['Start_Position'].astype(int)
    dbscSNV['Chromosome'] = dbscSNV['Chromosome'].astype(str)
    dbscSNV['Start_Position'] = dbscSNV['Start_Position'].astype(int)

    # merge the input dataframe and dbscSNV data on the the chr,start,ref,alt columns, adding the ada_score and rf_score columns
    input_data = pd.merge(input_data, dbscSNV, 
                        on=['Chromosome', 'Start_Position', 'Reference_Allele',  'Alternate_Allele'], how='left')
        
    return input_data

def annotate(input_data):

    # run the myvariant annotator
    str_myvariant = "python3 myvariant_annotator.py --input_maf " + input_maf + " --output_maf " + myVariant_merged
    if debug:
        str_myvariant += " --debug"
    debug_print("\nRan the following command: " + str_myvariant, debug)
    os.system(str_myvariant)

    # get the dbscSNV annotations (or call the annotator)
    dbscSNV = get_dbscCNV_annotations()

    # run the oncoKB annotator
    oncokb_in, oncokb_out  = get_oncokb_annotations(input_maf, input_data)

    # merge the two annotation files created
    merged_maf = join_annotations(oncokb_in, oncokb_out, input_data, dbscSNV)
    
    debug_print(annotated_maf + " shape before dropping duplicates is " + str(merged_maf.shape), debug)
    
    merged_maf = merged_maf.drop_duplicates()

    if debug:
        merged_maf.to_csv(annotated_maf.replace(".maf", "_extended.maf"), sep = '\t', index = None)

    debug_print(annotated_maf + " shape after dropping duplicates is " + str(merged_maf.shape), debug)

    return merged_maf

def Calculate_MAF(merged_data):
    if 'Normal_Sample' not in merged_data:
        return merged_data

    print("\nCalculate Minor Allele Frequency (MAF)")

    # drop duplicate values and make a column from the last 
    sample_list = merged_data[['Normal_Sample']].drop_duplicates()
    sample_list['Panel_type'] = sample_list['Normal_Sample'].str.split("-").str.get(-1)

    sample_count_panel = pd.DataFrame(
        {'sample_count_panel' : 
                            sample_list.groupby(['Panel_type'])['Normal_Sample'].nunique()
        }).reset_index()

    dict_panel_count = dict(list(zip(sample_count_panel['Panel_type'], sample_count_panel['sample_count_panel'])))
    expected_keys = ['IM3', 'IM5', 'IM6']
    for key in expected_keys:
        if key not in list(dict_panel_count.keys()):
            dict_panel_count[key] = 0

    print(dict_panel_count)

    merged_data['n_alt_freq'] = merged_data['n_alt_count']/merged_data['n_depth']
    merged_data["allele_count"] =  np.where(merged_data['n_alt_freq']>0.75, 2, 1)
    merged_data['Hugo_Symbol'] = merged_data['Hugo_Symbol'].fillna(".")

    MAF_alterations = pd.DataFrame({
            'minor_allele_count' : 
            merged_data.groupby(['Chromosome', 'Start_Position', 'End_Position', 'Reference_Allele', 'Alternate_Allele',
                                    'Hugo_Symbol'
                                ])['allele_count'].sum(),
            'median_VAF' : 
            merged_data.groupby(['Chromosome', 'Start_Position', 'End_Position', 'Reference_Allele', 'Alternate_Allele',
                                    'Hugo_Symbol'
                                ])['n_alt_freq'].median()
            
        }).reset_index()
    debug_print("MAF_alterations shape is " + str(MAF_alterations.shape), debug)
    debug_print("dataframe in MAF Function shape is " + str(merged_data.shape), debug)
    MAF_alterations['sample_count_bypanel'] = dict_panel_count['IM3']+dict_panel_count['IM5']+dict_panel_count['IM6']

    #adjust MAF calculation for genes that are only in CV5 or CV6
    MAF_alterations['sample_count_bypanel'] = np.where(MAF_alterations['Hugo_Symbol'].isin(cv6), 
                                                        dict_panel_count['IM6'], 
                                                        MAF_alterations['sample_count_bypanel']
                                                        )
    MAF_alterations['sample_count_bypanel'] = np.where(MAF_alterations['Hugo_Symbol'].isin(cv5), 
                                                        dict_panel_count['IM6'] + dict_panel_count['IM5'], 
                                                        MAF_alterations['sample_count_bypanel']
                                                        )
    MAF_alterations['sample_count_bypanel'] = MAF_alterations['sample_count_bypanel'].astype(float)
    MAF_alterations['MinorAlleleFreq'] = MAF_alterations['minor_allele_count']/(2* MAF_alterations['sample_count_bypanel'])
    MAF_alterations = MAF_alterations.sort_values('MinorAlleleFreq', ascending=False)
    MAF_alterations = MAF_alterations[['Chromosome', 'Start_Position', 'End_Position', 'Reference_Allele', 'Alternate_Allele',
                                        'median_VAF', 'MinorAlleleFreq' ]]

    # merge the MAF file with the merged_data on the common columns
    merged_maf = pd.merge(merged_data, MAF_alterations, 
                            on=['Chromosome', 'Start_Position', 'End_Position', 'Reference_Allele', 'Alternate_Allele',],
                            how='left')

    if debug:
        MAF_alterations.to_csv(input_maf.replace('.maf', '_maf_alterations.maf'), sep = '\t', index = None)

    return merged_maf

def get_gene_level_annotations():
	# get annotations for germline genes as oncogenes / tumor suppressors or genes whose function is via gain vs. loss
	gene_level_annotation = pd.read_csv(gl_input, sep = '\t', low_memory = False)
	gene_annotation_columns = gene_level_annotation.columns.tolist()
	gene_annotation_columns = gene_annotation_columns.remove('Hugo_Symbol')

	oncogenes = gene_level_annotation[gene_level_annotation['OncoKB Oncogene']==1]['Hugo_Symbol'].unique().tolist()
	oncogenes.append('TP53')
	#POLE and POLD1 are added as oncogenes because we know that only missense mutations in POLE lead to signature 10. 
	oncogenes.append('POLE')
	oncogenes.append('POLD1')
	non_cancergenes = gene_level_annotation[gene_level_annotation['OMIM']==0]['Hugo_Symbol'].unique().tolist()


	tumor_suppressors =  gene_level_annotation[gene_level_annotation['OncoKB TSG']==1]['Hugo_Symbol'].unique().tolist()
	tumor_suppressors.remove('POLE')
	tumor_suppressors.remove('POLD1')
	tumor_suppressors.append('EPCAM')

	#this file contains some additional gene annotations from CB.
	function_map_other_genes = pd.read_csv(gf_map_input, sep='\t', low_memory = False)
	other_gof_genes = function_map_other_genes[function_map_other_genes['oncogenic mechanism']=='gain-of-function']['Hugo_Symbol'].tolist()
	other_lof_genes = function_map_other_genes[function_map_other_genes['oncogenic mechanism']=='loss-of-function']['Hugo_Symbol'].tolist()
	#print gene_level_annotation.head()

	tumor_suppressors =list(set(tumor_suppressors + other_lof_genes))
	oncogenes =list(set(oncogenes + other_gof_genes+['VTCN1', 'YES1', 'XPO1', 'TRAF7']))
	print(sorted(oncogenes[:5]))
	print(sorted(tumor_suppressors[:5]))
	return oncogenes, tumor_suppressors, gene_level_annotation

def downstream_annotations(cohort_maf_uniq, oncogenes, tumor_suppressors):
    try:
        cohort_maf_uniq['ExAC2_AF_ASJ'] = np.where(cohort_maf_uniq['ExAC2_AF_ASJ']=='.',0, 
                                                    cohort_maf_uniq['ExAC2_AF_ASJ'])
    except TypeError:
        pass

    cohort_maf_uniq['ExAC2_AF_ASJ'] = cohort_maf_uniq['ExAC2_AF_ASJ'].fillna(0)
    cohort_maf_uniq['ExAC2_AF_ASJ'] = cohort_maf_uniq['ExAC2_AF_ASJ'].astype(float)
    cohort_maf_uniq['ExAC2_AF'] = np.where(pd.isnull(cohort_maf_uniq['ExAC2_AF']), 0, cohort_maf_uniq['ExAC2_AF'])
    cohort_maf_uniq['mutation_mechanism_consistency'] = 0

    cohort_maf_uniq['mutation_mechanism_consistency'] = np.where((cohort_maf_uniq['Variant_Classification'].isin([
        'Nonsense_Mutation', 'Frame_Shift_Ins', 'Frame_Shift_del', 'Splice_Site', 'Missense_Mutation', 
        'Splice_Region'
            ])) &
            (cohort_maf_uniq['Hugo_Symbol'].isin(tumor_suppressors)), 1,  
            cohort_maf_uniq['mutation_mechanism_consistency'] ) 

    cohort_maf_uniq['mutation_mechanism_consistency'] = np.where((cohort_maf_uniq['Variant_Classification'].isin([
        'Missense_Mutation'])) & (cohort_maf_uniq['Hugo_Symbol'].isin(oncogenes)), 1,  
            cohort_maf_uniq['mutation_mechanism_consistency'] ) 

    list_consequences =  cohort_maf_uniq['Consequence'].unique().tolist()
    list_consequences_uniq = []
    for i in list_consequences:
        list_items = i.split(",")
        list_consequences_uniq = list_consequences_uniq + list_items
    list_consequences_uniq = list(set(list_consequences_uniq))
    list_consequences_uniq = [i for i in list_consequences_uniq]
	
    for colname in list_consequences_uniq:
        cohort_maf_uniq['Consequence_'+colname] = np.where(
            cohort_maf_uniq['Consequence'].str.find(colname)>-1, 1, 0)
    list_consequences_uniq = ["Consequence_"+i for i in list_consequences_uniq]  

    splice_variants = cohort_maf_uniq.copy()[cohort_maf_uniq['Variant_Classification'].isin(['Splice_Site','Splice_Region'])]
    non_splice_variants = cohort_maf_uniq.copy()[~(cohort_maf_uniq['Variant_Classification'].isin(['Splice_Site','Splice_Region']))]
    non_splice_variants['splice_dist'] = -200

    splice_variants['temp'] = splice_variants['HGVSc']
    splice_variants['temp'] = splice_variants['temp'].str.replace("+",";", regex = False)
    splice_variants['temp'] = splice_variants['temp'].str.replace("-",";", regex = False)

    splice_variants['temp1'] = splice_variants['temp'].str.split(";").str.get(1)
    splice_variants['splice_dist'] = splice_variants['temp1'].str.extract('(\d+)')
    splice_variants['splice_dist'] = splice_variants['splice_dist'].fillna(0)

    splice_variants = splice_variants.drop(['temp', 'temp1'], axis =1 )

    cohort_maf_uniq = pd.concat([splice_variants, non_splice_variants])
    debug_print("dataframe in downstream annotations shape is " + str(cohort_maf_uniq.shape), debug)

    cohort_maf_uniq['splice_dist'] = cohort_maf_uniq['splice_dist'].astype(int)

    cohort_maf_uniq['mutation'] = cohort_maf_uniq['Hugo_Symbol'] + ":" + cohort_maf_uniq['HGVSc']
    cohort_maf_uniq['ratio_ASJ'] = cohort_maf_uniq['ExAC2_AF_ASJ']/cohort_maf_uniq['ExAC2_AF'].astype(float)
    cohort_maf_uniq['ratio_ASJ'] = cohort_maf_uniq['ratio_ASJ'].astype(float)
    cohort_maf_uniq['ratio_ASJ'] = cohort_maf_uniq['ratio_ASJ'].fillna(0.0)
    cohort_maf_uniq['ExAC2_AF'] = cohort_maf_uniq['ExAC2_AF'].fillna(0.0)

    pathogenic_terms = ['Pathogenic', 'pathogenic', ]
    benign_terms = ['Benign', 'benign']
    uncertain_terms = ['conflicting'] 

    #makes new column that prescribes pathogenicity if the CLINICAL_SIGNIFICANCE column contains the word pathogenic (which includes: Conflicting_interpretations_of_pathogenicity)
    cohort_maf_uniq['clinvar_pathogenic'] = np.where(cohort_maf_uniq['CLINICAL_SIGNIFICANCE'].str.contains('|'.join(pathogenic_terms),na=False),
                                                1, 0)
    cohort_maf_uniq['clinvar_uncertain'] = np.where(cohort_maf_uniq['CLINICAL_SIGNIFICANCE'].str.contains('|'.join(uncertain_terms), na=False),
                                                1, 0)
    cohort_maf_uniq['clinvar_benign'] = np.where(cohort_maf_uniq['CLINICAL_SIGNIFICANCE'].str.contains('|'.join(benign_terms), na=False),
                                                1, 0)

    cohort_maf_uniq['is-a-hotspot'] = cohort_maf_uniq['is-a-hotspot'].fillna('N')
    try:
        cohort_maf_uniq['is-a-hotspot'] = np.where(cohort_maf_uniq['is-a-hotspot']=='Y', 1 , 0)
    except TypeError:
        cohort_maf_uniq['is-a-hotspot'] = 0

    cohort_maf_uniq['oncogenic'] = np.where(cohort_maf_uniq['oncogenic'].isin(['Oncogenic', 'Likely Oncogenic']), 1, 0)

    #annotate last exon - 5' end of exon 
    # if 'Exon_Number' in cohort_maf_uniq:
    cohort_maf_uniq['exon_number'] = cohort_maf_uniq['Exon_Number'].str.split('/').str.get(0)
    cohort_maf_uniq['total_exons'] = cohort_maf_uniq['Exon_Number'].str.split('/').str.get(1)
    cohort_maf_uniq['Protein_position'] = cohort_maf_uniq['Protein_position'].str.replace("\?-", "",  regex = False)

    #breakpoint()
    cohort_maf_uniq['protein_position'] = cohort_maf_uniq['Protein_position'].str.split('/').str.get(0)
    cohort_maf_uniq['transcript_len'] = cohort_maf_uniq['Protein_position'].str.split('/').str.get(1)

    cohort_maf_uniq['protein_position'] = np.where(cohort_maf_uniq['protein_position']=='-', 0, cohort_maf_uniq['protein_position'])
    cohort_maf_uniq['protein_position'] = np.where(cohort_maf_uniq['protein_position'].str.find('?')>-1,
                                                    cohort_maf_uniq['protein_position'].str.replace('?', '', regex = False), 
                                                                        cohort_maf_uniq['protein_position'])
    cohort_maf_uniq['protein_position'] = np.where(cohort_maf_uniq['protein_position'].str.find('-')>-1,
                                                    cohort_maf_uniq['protein_position'].str.split("-").str.get(0), 
                                                                        cohort_maf_uniq['protein_position'])


    cohort_maf_uniq['protein_position'] = cohort_maf_uniq['protein_position'].fillna(0)
    cohort_maf_uniq['transcript_len'] = cohort_maf_uniq['transcript_len'].fillna(0)

    cohort_maf_uniq['protein_position'] = cohort_maf_uniq['protein_position'].replace(r'^\s*$', 0, regex=True)
    cohort_maf_uniq['protein_position'] = cohort_maf_uniq['protein_position'].astype(int)
    cohort_maf_uniq['transcript_len'] = cohort_maf_uniq['transcript_len'].astype(int)
    cohort_maf_uniq['tail_end'] = np.where((cohort_maf_uniq['protein_position']/cohort_maf_uniq['transcript_len'])>0.95, 1, 0)

    cohort_maf_uniq['last_exon_terminal'] = np.where((cohort_maf_uniq['tail_end']==1) & 
                                                                        (cohort_maf_uniq['exon_number']==cohort_maf_uniq['total_exons']),
                                                    1, 0)

    print("The shape of %s is: %s" % (annotated_maf, str(cohort_maf_uniq.shape)))
    cohort_maf_uniq.to_csv(annotated_maf, sep = '\t', index = None)

# def main():
#     # parse arguments from command line and then initialize necessary directories
#     setup_globals()

#     # load the input data into dataframe
#     input_maf_data = pd.read_csv(input_maf, sep = '\t', low_memory = False)
#     debug_print("\nThe shape of the input file " + "(" + input_maf + ") is " + str(input_maf_data.shape), debug)

#     #remove black list variants
#     try:
#         input_maf_data = input_maf_data[~(input_maf_data['DMP_Blacklist']=='Blacklist')]
#     except TypeError:
#         pass
#     except KeyError:
#         pass

#     debug_print(annotated_maf + " shape is " + str(input_maf_data.shape), debug)

#     # create the MAF file and merge it with the annotated dataframe
#     input_maf_data = Calculate_MAF(input_maf_data)
    
#     debug_print(annotated_maf + " shape is " + str(input_maf_data.shape), debug)

#     # remove all values of ExAC2_AF greater than 2% frequency
#     input_maf_data_rare = input_maf_data[input_maf_data['ExAC2_AF']<0.02 ]

#     if debug:
#         input_maf_data_rare.to_csv(input_maf.replace('.maf', '_ExAC2_rare.maf'), sep = '\t', index = None)

#     # drop any duplicates introduced by MAF file
#     input_maf_data_uniq = input_maf_data_rare.drop_duplicates(subset=['Chromosome', 'Start_Position',
#                                                                         'End_Position', 'Reference_Allele',
#                                                                         'Alternate_Allele'])
#     debug_print(annotated_maf + " shape is " + str(input_maf_data_uniq.shape), debug)

#     if debug:
#         input_maf_data_uniq.to_csv(input_maf.replace('.maf', '_uniq.maf'), sep = '\t', index = None)

#     # annotate using MyVariantInfo, OncoKB, dbscSNV
#     merged_maf = annotate(input_maf_data_uniq)

#     debug_print(annotated_maf + " shape is " + str(merged_maf.shape), debug)

#     oncogenes, tumor_suppressors, gene_level_annotation = get_gene_level_annotations()
#     merged_maf = pd.merge(merged_maf, gene_level_annotation, on='Hugo_Symbol', how='left')

#     downstream_annotations(merged_maf, oncogenes, tumor_suppressors)


# if __name__ == '__main__':
# 	main()

def main():
    # parse arguments from command line and then initialize necessary directories
    setup_globals()

    # load the input data into dataframe
    input_maf_data = pd.read_csv(input_maf, sep = '\t', low_memory = False)
    debug_print("\nThe shape of the input file " + "(" + input_maf + ") is " + str(input_maf_data.shape), debug)

    # annotate using MyVariantInfo, OncoKB, dbscSNV
    merged_maf = annotate(input_maf_data)

    #remove black list variants
    try:
        merged_maf = merged_maf[~(merged_maf['DMP_Blacklist']=='Blacklist')]
    except TypeError:
        pass
    except KeyError:
        pass

    debug_print(annotated_maf + " shape is " + str(merged_maf.shape), debug)

    # create the MAF file and merge it with the annotated dataframe
    merged_maf = Calculate_MAF(merged_maf)
    
    debug_print(annotated_maf + " shape is " + str(merged_maf.shape), debug)

    # remove all values of ExAC2_AF greater than 2% frequency
    merged_maf_rare = merged_maf[merged_maf['ExAC2_AF']<0.02 ]

    if debug:
        merged_maf_rare.to_csv(input_maf.replace('.maf', '_ExAC2_rare.maf'), sep = '\t', index = None)

    # drop any duplicates introduced by MAF file
    merged_maf_uniq = merged_maf_rare.drop_duplicates(subset=['Chromosome', 'Start_Position',
                                                                        'End_Position', 'Reference_Allele',
                                                                        'Alternate_Allele'])
    debug_print(annotated_maf + " shape is " + str(merged_maf_uniq.shape), debug)

    if debug:
        merged_maf_uniq.to_csv(input_maf.replace('.maf', '_uniq.maf'), sep = '\t', index = None)

    oncogenes, tumor_suppressors, gene_level_annotation = get_gene_level_annotations()
    merged_maf_uniq = pd.merge(merged_maf_uniq, gene_level_annotation, on='Hugo_Symbol', how='left')

    downstream_annotations(merged_maf_uniq, oncogenes, tumor_suppressors)


if __name__ == '__main__':
	main()