import datetime, os, pprint, re, sys, time, matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import pandas as pd
from colour import Color
import matplotlib.style as style


def plot_figure(name, data, label_name, path, ben):

    style.use('seaborn-poster')
    matplotlib.rcParams['font.family'] = "sans-serif"
    fig, ax = plt.subplots()

    newdic = dict(data.Variant_Classification.value_counts())

    pathogenic_dict = dict(data.groupby(label_name)['Variant_Classification'].value_counts().loc[path])
    benign_dict = dict(data.groupby(label_name)['Variant_Classification'].value_counts().loc[ben])

    for key in newdic:
        if key not in benign_dict:
            benign_dict[key] = 0
        if key not in pathogenic_dict:
            pathogenic_dict[key] = 0     

    x = [label.replace('_',' ') for label in newdic.keys()]
    
    y_path = []
    y_ben = []
    for key in newdic.keys():
        y_path.append(pathogenic_dict[key])
        y_ben.append(benign_dict[key])

    # y_path = pathogenic_dict.values()
    # y_ben = benign_dict.values()

    # red = Color("red")
    # colors = list(red.range_to(Color("green"),len(x)))
    # colors = [color.rgb for color in colors]

    p1 = ax.bar(x, y_path, color='tab:orange', label = 'Pathogenic')
    p2 = ax.bar(x, y_ben, bottom=y_path, color='tab:blue', label = 'Benign')


    # v = plt.bar(x, y, color=colors)
    plt.xlabel('Mutation Types', labelpad=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Count', labelpad=12)
    plt.yscale("log")
    plt.tight_layout()
    plt.title("Distribution of Mutations: %s Training Data" % name.capitalize())
    plt.legend()

    # for r1,r2 in zip(p1,p2):
    #     h1 = r1.get_height()
    #     h2 = r2.get_height()
    #     plt.text(r1.get_x()+r1.get_width()/2., h1+h2, '%s'% (h1+h2), ha = 'center', va='bottom')

    for p in p1.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy() 
        if height > 1 and height < 100:
            ax.annotate(height, (p.get_x()+width*0.35, p.get_y()+height-0.3*height), 
                        fontsize=12, color='black')
        if height== 1:
            ax.annotate(height, (p.get_x()+width*0.35, p.get_y()+height), 
                        fontsize=12, color='black')
        if height >= 100:
            ax.annotate(height, (p.get_x()+width*0.2, p.get_y()+height-0.3*height), 
                        fontsize=12, color='black')

    for p in p2.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy() 
        if height > 0 and height < 100:
            ax.annotate(height, (p.get_x()+width*0.35, p.get_y()+height), 
                        fontsize=12,  color='black')
        if height >= 100:
            ax.annotate(height, (p.get_x()+width*0.2, p.get_y()+height), 
                        fontsize=12,  color='black')

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)

    fig.savefig('%s_training_dist_bar.jpg'%name.capitalize(), dpi = 1500, bbox_inches = 'tight')

def main():
    old = pd.read_csv('../input_files/classifier_training_data_V1.txt.gz', compression = 'gzip', sep = '\t', low_memory=False)
    plot_figure("old", old, "signed_out", 1, 0)

    # new = pd.read_csv('../input_files/classifier_training_data_V2.maf.gz', compression = 'gzip', sep = '\t', low_memory=False)
    # plot_figure("new", new, "pathogenic", True, False)

    old_new_combined = pd.read_csv('../input_files/classifier_training_data_V1_V2.maf.gz', compression = 'gzip', sep = '\t', low_memory=False)
    plot_figure("new", old_new_combined, "pathogenic", 1, 0)

if __name__ == '__main__':
	main()	