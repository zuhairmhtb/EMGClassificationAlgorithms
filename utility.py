import numpy as np
from prettytable import PrettyTable, from_html_one
import os, sys, math

def html_to_latex_table(html_string, table_caption):
    pt = from_html_one(html_string)
    latex_table_syntax = ["\\begin{table}[h]\n\\centering", "\\end{table}"]
    latex_caption = "\\caption{" + str(table_caption) + "}"
    latex_row_size = ""
    for i in range(len(pt.field_names)):
        latex_row_size += "1"
        if i != len(pt.field_names)-1:
            latex_row_size += " "

    latex_table_begin_syntax = ["\\begin{tabular}{" + latex_row_size + "}", "\\end{tabular}"]
    latex_hline = "\\hline"
    latex_table_header = ""
    for i in range(len(pt.field_names)):
        latex_table_header += "\\textbf{" + pt.field_names[i] + "}"
        if i != len(pt.field_names)-1:
            latex_table_header += " & "
        else:
            latex_table_header += "\\\\"
    latex_table_data = ""
    for i in range(len(pt._rows)):
        td = pt._rows[i]
        td_str = ""
        for j in range(len(td)):
            td_str += str(td[j])
            if j != len(td) - 1:
                td_str += " & "
            else:
                td_str += " \\\\"
        latex_table_data = latex_table_data + td_str + "\n"

    latex_result = latex_table_syntax[0] + "\n" + \
                   latex_table_begin_syntax[0] + "\n" + \
                   latex_hline + "\n" \
                   + latex_table_header + "\n" + \
                   latex_hline + "\n" + \
                   latex_table_data + \
                   latex_hline + "\n" + \
                   latex_table_begin_syntax[1] + "\n" + \
                   latex_caption + "\n" + \
                   latex_table_syntax[1]
    return latex_result


dir = 'D:\\thesis\\ConvNet\\MyNet\\emg_classification_library\\time_freq_classification_output\\spectral_peaks_table.html'
with open(dir, 'r') as fp:
    html_string = fp.read()
    latex_result = html_to_latex_table(html_string, "Spectral Peak Amplitude and Frequency from Fast Fourier Transform")
    print(latex_result)