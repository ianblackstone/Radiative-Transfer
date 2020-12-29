import csv

full_data_set = 'Sil_81/Sil_81'

with open(full_data_set) as Draine:
    data = Draine.read()
    
data = data.split("\n\n")

del data[0:1]

for data_set in data:
    grain_size = data_set.split(" ",1)[0]
    
    file_name = 'draine_data_Sil_' + str(float(grain_size)).replace('.','_')
    
    data_set = data_set.split('\n')
    
    del data_set[0]
    del data_set[-1]
    
    output_file = open(file_name,"w")
    
    for line in data_set:
        output_file.write(line + '\n')
    
    output_file.close()