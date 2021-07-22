# Germline-Classifier
This is the source code for the random forest based classifier described in Srinivasan P, Bandlamudi C, et al. In preparation. This was used to classify rare germline variants called across 17,152 advanced cancer patients sequenced using the MSK-IMPACT assay.

Requirements:
Refer to requirements.txt to find other python dependencies including myvariant python package from https://docs.myvariant.info/en/latest/doc/packages.html#myvariant-python-module

Inputs to Scripts :
To run the classifier, you will need an input file in maf format with the following columns :

#Variant level info

Chromosome, Start_Position,End_Position, Reference_Allele, Alternate_Allele, Variant_Type, Variant_Classification,Consequence, 
Hugo_Symbol, Protein_position, HGVSp_Short, Exon_Number

#Patient level info

Normal_Sample, n_alt_count, n_depth, 

#annotations from Clinvar and gnomAD

ExAC2_AF, ExAC2_AF_ASJ, CLINICAL_SIGNIFICANCE, GOLD_STARS,  

If you don't have variant level and ClinVar, gnomAD annotations, please perform preliminary annotation with VariantEffectPredictor
The coordinates are expected to conform to VEP standards in order to allow for correct merging with curated training data

# Usage

Step 1: python3 preprocess.py --input_maf [/path/to/input_file_location] --annotated_maf [/path/to/desired/output_file_location] --scripts_dir [/path/to/annotation_files/and/oncokb_annotator] [OPTIONAL:--debug]

This step adds additional annotations from myvariant.info, OncoKB and dbScSNV and uses some of this information to create more annotations (Ex. MAF).
  
Step 2: python3 germline_classifier.py --classifier_input [/path/to/results/from/preprocess.py] --classifier_output [/path/to/desired/classifier_output_location] --scripts_dir [/path/to/gene_annotation_data] --training_data [/path/to/training_data] --type [space separated values from type_list**] --features [/path/to/features_to_keep_file]

  This will train the Random Forest model on the provided training data and perform predictions on the output file.

Test data is provided in input_data folder. 

** type_list = ['Missense', 'Splice', 'Truncating', 'Misc','In_Frame', 'Silent',  "UTR", "Intron", "IGR", "RNA"]

# Example

Note: ** This assumes you are in the folder containing the python scripts **

#create all relevant columns and annotations
python3 preprocess.py --input_maf ../input_files/test_input.maf --annotated_maf ../output_files/annotated_test_input.maf --scripts_dir ..

#run classifier
python3 germline_pathogenicity_classifier.py --classifier_input ../output_files/annotated_test_input.maf --classifier_output ../output_files/classificer_out_test_input.maf --scripts_dir

# Output
Predictions from classifier are stored in column "prediction" as a binary value. The column "prediction_probability" shows the probability of being called pathogenic for every variant. 

Note: the scripts extract rare variants based on gnomAD frequencies and minor allele frequency within provided aggregate maf. It also extracts unique variants and subsets to variant classes that exclude UTR and silent mutations.  


