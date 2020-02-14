import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
import csv

def preprocess(data):
    """
    Reduces ICD9 codes to level 3
    param data: numpy array containing ICD9 codes
    """
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i,j] != 'nan':
                data[i,j] = data[i,j][:3]
    return data

def create_dict(data):
    """
    Creates mapping and inverse mapping from ICD9 codes to unique ids
    param data: preprocessed numpy array containing ICD9 codes 
    """
    icd_to_id = {}
    id_to_icd = {}
    count = 0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i,j] != 'nan':
                if data[i,j] not in icd_to_id.keys():
                    count += 1
                    icd_to_id[data[i,j]] = count
                    id_to_icd[count] = data[i,j]
    return icd_to_id,id_to_icd

def extract_counts_mat(data,icd_to_id,id_to_icd):
    """
    Extracts counts D_ij for every pair of diseases i and j
    param data: preprocessed numpy array containing ICD9 codes 
    """
    N = len(list(icd_to_id.keys()))
    count_mat = np.zeros((N,N))
    for i in range(data.shape[0]):
        for j in range(2,data.shape[1]):
            if data[i,j] == 'nan':
                break
            if icd_to_id[data[i,0]] != icd_to_id[data[i,j]]:
                count_mat[icd_to_id[data[i,0]] - 1,icd_to_id[data[i,j]] - 1] += 1
    return count_mat


#def extract_counts(data,icd_to_id,id_to_icd):
#    counts_dict = {}
#    for i in range(data.shape[0]):
#        if icd_to_id[data[i,0]] not in counts_dict.keys():
#            counts_dict[icd_to_id[data[i,0]]] = {}
#        for j in range(2,data.shape[1]):
#            if data[i,j] == 'nan':
#                break
#            if icd_to_id[data[i,j]] not in counts_dict[icd_to_id[data[i,0]]].keys():
#                counts_dict[icd_to_id[data[i,0]]][icd_to_id[data[i,j]]] = 1
#            else:
#                counts_dict[icd_to_id[data[i,0]]][icd_to_id[data[i,j]]] += 1
#    return counts_dict


def calculate_RR_and_pvalue(count_mat):
    """
    Calculates the Relative Risk and pvalue from Fisher's exact test for every pair of diseases i and j
    param count_mat: numpy array containing counts for every pair of diseases i and j
    """
    N = count_mat.shape[0]
    count_mat_sum = count_mat.sum()
    count_mat_row_sum = count_mat.sum(axis=1)
    count_mat_col_sum = count_mat.sum(axis=0)

    RR_mat = np.zeros(count_mat.shape)
    pval_mat = np.zeros(count_mat.shape)
    for i in range(N):
        for j in range(N):
            if(i!=j and count_mat_col_sum[j] > 0.0 and count_mat_row_sum[i] > 0.0):
                mat = np.zeros((2,2))
                mat[0,0] = count_mat[i,j]
                mat[0,1] = count_mat_row_sum[i] - count_mat[i,j]
                mat[1,0] = count_mat_col_sum[j] - count_mat[i,j]
                mat[1,1] = count_mat_sum - count_mat_row_sum[i] - count_mat_col_sum[j] + count_mat[i,j]
                RR_mat[i,j] = (count_mat[i,j] * count_mat_sum) / (count_mat_row_sum[i] * count_mat_col_sum[j])
                _,pval = fisher_exact(mat,alternative='greater')
                pval_mat[i,j] = pval
    return RR_mat,pval_mat


def write_output(RR_mat,pval_mat,R_thresh,p_thresh,id_to_icd):
    """
    Determines if an edge is significant based on Relative Risk and pvalue and writes all significant edges to csv file
    param RR_mat: numpy array containing Relative risk values for every pair of diseases i and j
    param pval_mat: numpy array containing pvalues for every pair of diseases i and j
    param R_thresh: Threshold for Relative risk
    param p_thresh: Threshold for pvalue
    param id_to_icd: mapping from id to ICD9 codes
    """
    f = open('output.csv','w')
    writer = csv.writer(f)
    N = RR_mat.shape[0]
    writer.writerow(['prior diagnosis','later diagnosis','count','Relative Risk','p-value'])
    for i in range(N):
        for j in range(N):
            if(i!=j and RR_mat[i,j] >= R_thresh and pval_mat[i,j] <= p_thresh):
                #print(id_to_icd[i+1],id_to_icd[j+1])
                writer.writerow([id_to_icd[i+1],id_to_icd[j+1],count_mat[i,j],RR_mat[i,j],pval_mat[i,j]])
    f.close()
    return


FILE = '/home/gururaj/CORMAC_AI_CHALLENGE/DATA/2008/inp_clm_saf_lds_5_2008.csv'
df = pd.read_csv(FILE,usecols=[1,22,83,84,85,86,87,88,89,90,91,92],dtype=str,header=None)

data = np.array(df.values[:,1:],dtype=str)

data = preprocess(data)
icd_to_id,id_to_icd = create_dict(data)
#counts_dict = extract_counts(data,icd_to_id,id_to_icd)
count_mat = extract_counts_mat(data,icd_to_id,id_to_icd)
RR_mat,pval_mat = calculate_RR_and_pvalue(count_mat)
write_output(RR_mat,pval_mat,4,0.0001,id_to_icd)
