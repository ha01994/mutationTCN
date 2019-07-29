import pandas as pd
import numpy as np 

#Drop columns that are not part of the alignment
def prune_seq(sequence):
    output=""
    for s in sequence:
        if s!="." and not (s.islower()):
            output+=s
    return output

#Find the indices of aligned columns, Note that indices are 0-indexed
def index_of_non_lower_case_dot(sequence):
    output=[]
    for s in range(len(sequence)):
        if sequence[s]!="." and not (sequence[s].islower()):
            output.append(s)
    return output

#Helper function to translate string to one_hot
def translate_string_to_one_hot(sequence,order_list):
    out=np.zeros((len(order_list),len(sequence)))
    for i in range(len(sequence)):
        out[order_list.index(sequence[i])][i]=1
    return out

#generate single mutants for those positions that experimental data and alignment are available
def mutate_single(wt,mutation_data,offset=0,index=0):
    mutants=[]
    prev=int(mutation_data[0][1])-offset

    for md in mutation_data:
        if prev!=int(md[1])-offset:
           index+=1
           prev=int(md[1])-offset 
        mutant=[md[2] if i==index else wt[i] for i in range (len(wt))]
        mutants.append(mutant)
        
    return mutants

#generate double mutants for those positions that experimental data and alignment are available
def mutate_double(wt,mutation_data1,mutation_data2,offset=0,index=0):
    mutants=[]
    mutants_double=[]
    index2=int(mutation_data2[0][1])-int(mutation_data1[0][1])+index

    #make first mutation
    prev=int(mutation_data1[0][1])-offset
    for md in mutation_data1:
        if prev!=int(md[1])-offset: #new
           index+=int(md[1])-offset-prev
           prev=int(md[1])-offset 
        mutant=[md[2] if i==index else wt[i] for i in range (len(wt))]
        mutants.append(mutant)
    
    #make second mutation
    prev=int(mutation_data2[0][1])-offset
    for md,mutant in zip(mutation_data2,mutants):
        if prev!=int(md[1])-offset: #new
           index2+=int(md[1])-offset-prev
           prev=int(md[1])-offset 
        mutant=[md[2] if i==index2 else mutant[i] for i in range (len(mutant))]
        mutants_double.append(mutant)
        
        
    return mutants_double

#generate a pandas dataframe from an alignment file
def pdataframe_from_alignment_file(filename,num_reads=200000):
    data=pd.DataFrame(columns=["name","sequence"])
    with open(filename) as datafile:
        serotype=""
        sequence=""
        dump=False
        count=0
        dataf=datafile.readlines()
        for line in dataf:
            if line.startswith(">"):
               if count>=num_reads:
                  break
               if dump:
                  row=pd.DataFrame([[serotype,sequence]],columns=["name","sequence"])
                  data=data.append(row,ignore_index=True)
               dump=True
               serotype=line[1:].strip("\n")
               sequence=""
               count+=1
            else:
                sequence+=line.strip("\n")
        row=pd.DataFrame([[serotype,sequence]],columns=["name","sequence"])
        data=data.append(row,ignore_index=True)
    return data

#compute distance between two aligned sequences
def aligned_dist(s1,s2):
    count=0
    for i,j in zip(s1,s2):
        if i!=j:
            count+=1
    return count

#Compute a new weight for sequence based on similarity threshold theta 
def reweight_sequences(dataset,theta):
    weights=[1.0 for i in range(len(dataset))]
    for i in range(len(dataset)):
        for j in range(i+1,len(dataset)):
            if aligned_dist(dataset[i],dataset[j])*1./len(dataset[i]) <theta:
               weights[i]+=1
               weights[j]+=1
    return list(map(lambda x:1./x, weights))



