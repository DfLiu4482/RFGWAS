#处理VCF文件
import csv
import numpy as np

with open('Neogen_China_POR80KV01_20170813.vcf', 'r', encoding='utf-8') as file:
    csvfile = open('Neogen_China_POR80KV01_20170813.csv', 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(csvfile)
    for line in file:
        if not line.startswith('##'):
            line = line.replace('./.', '0')
            line = line.replace('0/0', '0')
            line = line.replace('0/1', '1')
            line = line.replace('1/1', '2')
            line = line.replace('\n', '')

            line_arr = line.split('\t')
            print(line_arr)
            arr = line_arr[9:]
            final_arr = np.append(line_arr[2],arr)
            csv_writer.writerow(final_arr)
    csvfile.close()

import pandas as pd
phe = pd.read_excel('imf_phenotype_figshare.xlsx')
ffid = phe['FFID']
id = phe['ID']
ffid_str = ffid.astype(str)
id_str = id.astype(str)
merged_data = ffid_str + '_' + id_str
carcass_weight = phe['carcass_weight']
data = pd.concat([merged_data,carcass_weight],axis=1)
data.to_csv('imf_phenotype_figshare.csv', index=False)





