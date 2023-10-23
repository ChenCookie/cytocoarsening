# some of the MELD experiement borrow from MELD original paper(https://andreasloukas.blog/2018/11/05/multilevel-graph-coarsening-with-spectral-and-cut-guarantees/)).


import os
import csv
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.neighbors import kneighbors_graph
from graph_coarsening.coarsening_utils import *
import numpy as np
import matplotlib.pylab as plt

import networkx as nx
import pygsp as gsp
from pygsp import graphs
gsp.plotting.BACKEND = 'matplotlib'
import time
import seaborn as sns
import random
import meld
import graphtools as gt

def select_plot_data(part_patients,data2d,label_age,cell_number,ndcols):
    select_data2d=np.zeros((len(part_patients)*cell_number,ndcols))
    select_label_age=np.zeros(len(part_patients)*cell_number)
    for i in range(0,len(part_patients),1):
        for j in range(0,cell_number,1):
            for k in range(0,ndcols,1):
                select_data2d[(i*cell_number)+j][k]=data2d[(part_patients[i]*cell_number)+j][k]
            select_label_age[(i*cell_number)+j]=label_age[(part_patients[i]*cell_number)+j]
    return select_data2d,select_label_age


def load_file(path,file_name_list,pick_patient,pick_cell,feature_num):
    groupdata = np.empty([0, feature_num+1], dtype=float)
    length_of_index=list()
    if ".DS_Store" in file_name_list:
        file_name_list.remove(".DS_Store")
    for filename in range(0,pick_patient,1):
        with open(path+file_name_list[filename], "r", encoding = 'utf-8') as reference:
            reference_reader = csv.DictReader(reference, delimiter=',')
            df = pd.DataFrame(data=reference_reader)
            df = df.astype(float)
            randomlist = random.sample(range(0, len(df)), pick_cell)
            df_fix=np.array(df)[np.array(randomlist)]
            file_header=df.columns
            groupdata=np.concatenate((groupdata, df_fix), axis=0)
            length_of_index.append(len(df_fix))
    return groupdata,length_of_index,file_header

def load_pree_data(patientnum,cell_number,ndcols):
    arr = next(os.walk('/playpen-ssd/chijane/NK_cell_preprocess/'))[2]
    path='/playpen-ssd/chijane/NK_cell_preprocess/'
    dataWard,ward_length,file_header=load_file(path,arr,len(arr),cell_number,ndcols)
    labelC=dataWard[:, -1]
    dataWard=dataWard[:, :-1]
    dataC = np.array(dataWard)
    getpatient=list()
    for id in range(patientnum):
        subpatient=[id for i in range(cell_number)]
        getpatient+=subpatient
    return dataC,labelC,file_header,getpatient

sns.set(rc={"figure.figsize":(16, 12)})

def trace_node_label_form_dict(groups_dict,labelC,pass_num,k,dataC_idx,acc_num,err_num):
    get_remain_dix=list(set(dataC_idx).difference(list(map(int, groups_dict.keys()))))
    trace_num,trace_den=0,0
    trace_different=0
    node_num=0
    for i in groups_dict.keys():
        for j in groups_dict[i]:
            if j!= -1:
                trace_den+=1
                trace_different+=abs(labelC[j]-labelC[int(i)])
                if labelC[j]==labelC[int(i)]:
                    trace_num+=1
        node_num+=1
    for i in get_remain_dix:
        trace_den+=1
        trace_num+=1
        node_num+=1
    acc_num.append(trace_num/trace_den*100)
    err_num.append(trace_different/trace_den)
    return acc_num,err_num
    
def get_center_point_v2(subdata_33d,subgroups,sublabel):
    weighted=[0] * len(subdata_33d)
    subdata_mean=np.mean(subdata_33d, axis=0)
    dists=list()
    for i in range(0,len(subdata_33d),1):
        out_arr = np.subtract(subdata_mean, subdata_33d[i])
        out_arr=np.absolute(out_arr)
        dists.append(np.sum(out_arr))
    count_weight=len(subdata_33d)
    count_weight_denominator=(1+len(subdata_33d))*len(subdata_33d)/2
    for i in np.argsort(dists):
        weighted[i]+=(count_weight/count_weight_denominator)
        count_weight-=1
    sublabelcount=dict()
    for i in sublabel:
        if i in sublabelcount:
            sublabelcount[i]+=1
        else:
            sublabelcount[i]=1
    #print(weighted,subgroups,sublabelcount)
    sublabelcount=dict(sorted(sublabelcount.items(), key=lambda item: item[1]))
    all_weight_count=0
    alpha=1
    for key, value in sublabelcount.items():
        all_weight_count+=(value*value)
    for key, value in sublabelcount.items():
        for i in range(len(sublabel)):
            if sublabel[i]==key:
                weighted[i]+=(alpha*value/all_weight_count)
    #print(weighted,subgroups,sublabel)
    the_min_dix=weighted.index(max(weighted))
    return subgroups[the_min_dix]


def compute_std(store_edge,labelC,dataC_idx):
    final_std,std_denominator=0,0
    for i_sn in dataC_idx:
        find_edge=np.where(store_edge==i_sn)
        print(i_sn,find_edge)
        for in_find_edge in range(0,len(find_edge[0]),1):
            se_dim1=find_edge[0][in_find_edge]
            se_dim2=1-find_edge[1][in_find_edge]
            adjacent_node=store_edge[se_dim1][se_dim2]
            print("self",i_sn,labelC[i_sn])
            print("add",adjacent_node,labelC[adjacent_node])
            final_std+=(labelC[i_sn]-labelC[adjacent_node])**2
            std_denominator+=1
        final_std=final_std/std_denominator
    print("final_std",final_std,std_denominator)    
    return final_std

def build_graph(graph_idx,graph_edge):
    graph = nx.Graph()
    for idx in np.sort(graph_idx):graph.add_node(idx)
    # print("build_graph node",graph.nodes())
    # print("graph_edge",graph_edge)
    for edge in graph_edge:graph.add_edge(edge[0], edge[1], weight=1)
    # print("build_graph L\n",nx.laplacian_matrix(graph).toarray())
    return graph

def get_p(graph,x):
#     print("get_p label",x)
    L=nx.laplacian_matrix(graph)
#     print(L.toarray())
    # print(x)
    # print(x.shape,L.shape)
    return x.T @ L @ x

def find_node_key(node_num,group_dict):
    node_key_loc=-1
    for group_nd in group_dict.keys():
        if node_num in group_dict[group_nd]:
            node_key_loc=group_nd
            break
    return node_key_loc

def create_adjaency_matrix(input_idx_list,input_store_edge):
    adj_matrix=np.zeros((len(input_idx_list),len(input_idx_list)), dtype=float)
    count_edge=0
    for edge_pire in input_store_edge:
        matrixX=input_idx_list.index(edge_pire[0])
        matrixY=input_idx_list.index(edge_pire[1])
        adj_matrix[matrixX][matrixY]=1
        adj_matrix[matrixY][matrixX]=1
        count_edge+=1
    print(count_edge)
    return adj_matrix

def get_MELD_condition(input_labelC):
    meld_condition=list()
    for labels in input_labelC:
        # if labels==1:
        #     meld_condition.append("healthy")
        if labels==0:
            meld_condition.append("condition_0")
        elif labels==1:
            meld_condition.append("condition_1")
    return meld_condition


def get_ori_adjaency_matrix(input_dataC, set_k):
    A_KNN=kneighbors_graph(input_dataC, set_k,include_self=True, mode="distance")
    A_KNN= np.array(A_KNN.toarray())
    A_KNN=A_KNN+A_KNN.T
    A_KNN/=2

    ori_adjaency_matrix=np.zeros((len(A_KNN),len(A_KNN)), dtype=float)
    for i in range(len(A_KNN)):
        for j in range(len(A_KNN[0])):
            if A_KNN[i][j]>0:
                ori_adjaency_matrix[i][j]=1
    return ori_adjaency_matrix
def node_match_supernode(input_dataC,input_total_new_groups,input_idx_list):
    node_match_list=list()
    for i in range(0,len(input_dataC),1):
        data_loc=find_node_key(i,input_total_new_groups)
        node_match_list.append(input_idx_list.index(data_loc))
    return node_match_list


def get_MELD_score(input_adjacency_matrix,input_beta,condition_statue):
    g = gt.Graph(input_adjacency_matrix, precomputed="adjacency")
    meld_op = meld.MELD(beta=input_beta)
    sample_densities = meld_op.fit_transform(g, sample_labels=np.array(condition_statue))
    print("finish compute graph MELD score")
    return sample_densities

def get_prid_corre(input_mul_likelihoods,input_ori_likelihoods,input_mul_likelihoods_to_ori,input_condition,new_groups,input_labelC,input_idx_list):
    meld_label=[ 'condition_1' if input_mul_likelihoods['condition_1'][x]>0.5 else 'condition_0' for x in range(0, len(input_mul_likelihoods['condition_0'])) ]
    meld_ori_label=[ 'condition_1' if input_ori_likelihoods['condition_1'][x]>0.5 else 'condition_0' for x in range(0, len(input_ori_likelihoods['condition_0'])) ]

    predict_yes=0
    predict_ori_yes=0

    for i in range(0,len(input_ori_likelihoods['condition_0']),1):
        if input_condition[i]==meld_ori_label[i]:
            predict_ori_yes+=1

    for i in range(0,len(input_mul_likelihoods['condition_0']),1):
        if np.mean(input_labelC[new_groups[input_idx_list[i]]])<=0.5:
            coar_label='condition_0'
        else:
            coar_label='condition_1'
        if coar_label==meld_label[i]:
            predict_yes+=1

    graph_correlation=np.corrcoef(input_mul_likelihoods_to_ori['condition_0'], input_ori_likelihoods['condition_0'])[0][1]
    coar_prit=predict_yes/len(input_mul_likelihoods['condition_0'])
    ori_prit=predict_ori_yes/len(input_ori_likelihoods['condition_0'])
    print(graph_correlation,"predict percent:",coar_prit,ori_prit)
    return graph_correlation,coar_prit,ori_prit



ndcols=29
patientnum=20
cell_number=1000#470
t1 = time.time()
k=5
subsample=2
# ratios=[0.075,0.125,0.16,0.19,0.21,0.23,0.24,0.25,0.27,0.28]
multipass_df = pd.read_pickle("NK_multipass_1000_new.pkl")

subsample_accuracy=list()
subsample_error=list()
subsample_qua_label_divnod=list()
subsample_qua_label_divedg=list()
subsample_qua_feature_divnod=list()
subsample_qua_feature_divedg=list()
subsample_node_number=list()
subsample_edge_number=list()

subsample_meld_correlation=list()
subsample_coarsening_predict=list()
subsample_original_predict=list()
subsample_runtime=list()

for subsample_time in range(0,subsample):
    total_accuracy=list()
    total_error=list()
    qua_label_divnod=list()
    qua_label_divedg=list()
    qua_feature_divnod=list()
    qua_feature_divedg=list()
    node_number=list()
    edge_number=list()

    meld_correlation=list()
    coarsening_predict=list()
    original_predict=list()
    total_time=list()
    print('subsample',subsample_time+1)
    ratios=list(1-np.array(multipass_df['subsample_node_number'][subsample_time])/(patientnum*cell_number))
    for r in ratios:
        dataC,labelC,features,patientID=load_pree_data(patientnum,cell_number,ndcols)
        t1 = time.time()
        A = kneighbors_graph(dataC, k, mode="distance")
        A = A.toarray()
        A = A + np.transpose(A)
        A = A/2

        #make graph for GSP
        G = graphs.Graph(A)
        # print('done with graph')
        labelC=np.array(labelC)
        methods='variation_edges'
        print(methods,r)
        C, Gc, Call, Gall = coarsen(G, K=k,r=r, method=methods) #change different method of MELD
        total_group_dict=dict()
        C_array=C.toarray()
        for i in C_array:
            coarsen_list_idx=np.where(i>0)[0]
            np_node=get_center_point_v2(dataC[coarsen_list_idx],coarsen_list_idx,labelC[coarsen_list_idx])
            total_group_dict[np_node]=coarsen_list_idx
        super_node_idx=[*total_group_dict]
        node_number.append(len(super_node_idx))

        graph_coarsening_edge = np.array(Gc.get_edge_list()[0:2])
        reshape_edge= np.array([ [0]*2 for i in range(len(graph_coarsening_edge[0]))])
        for i in range(len(graph_coarsening_edge[0])):
            reshape_edge[i][0]=graph_coarsening_edge[0][i]
            reshape_edge[i][1]=graph_coarsening_edge[1][i]
        reshape_edge.sort()

        total_accuracy,total_error=trace_node_label_form_dict(total_group_dict,labelC,0,0,super_node_idx,total_accuracy,total_error)


        #sort
        reshape_edge=np.sort(np.array(super_node_idx)[reshape_edge])
        idx_list=sorted(total_group_dict.keys())

        super_node_idx.sort()
        subgraph=build_graph(super_node_idx,reshape_edge)
        qua_result=get_p(subgraph,labelC[super_node_idx])
        qua_label_divnod.append(qua_result/len(super_node_idx))
        qua_label_divedg.append(qua_result/len(reshape_edge))
        edge_number.append(len(reshape_edge))

        euqation_result=[]
        for i in range(0,ndcols,1):
            euqation_result.append(get_p(subgraph,dataC[super_node_idx][:,i]))
        qua_feature_divnod.append((sum(euqation_result) / len(euqation_result))/len(super_node_idx))
        qua_feature_divedg.append((sum(euqation_result) / len(euqation_result))/len(reshape_edge))
        t2 = time.time()
        total_time.append(t2-t1)


    distance_matrix=create_adjaency_matrix(idx_list,reshape_edge)
    condition=get_MELD_condition(labelC)

    ori_distance_matrix=get_ori_adjaency_matrix(dataC, k)

    beta_selection=1
    ori_sample_densities=get_MELD_score(ori_distance_matrix,beta_selection,condition)
    mul_sample_densities=get_MELD_score(distance_matrix,beta_selection,np.array(condition)[idx_list])
    match_back_data=node_match_supernode(dataC,total_group_dict,idx_list)
    chd_likelihoods = []
    mul_likelihoods=meld.normalize_densities(mul_sample_densities)
    ori_likelihoods=meld.normalize_densities(ori_sample_densities)
    mul_likelihoods_to_ori=mul_likelihoods.loc[ match_back_data , : ]
    meld_correlation,prid_c,prid_o=get_prid_corre(mul_likelihoods,ori_likelihoods,mul_likelihoods_to_ori,condition,total_group_dict,labelC,idx_list)

    subsample_meld_correlation.append(meld_correlation)
    subsample_coarsening_predict.append(prid_c)
    subsample_original_predict.append(prid_o)

    subsample_accuracy.append(total_accuracy)
    subsample_error.append(total_error)
    subsample_qua_label_divnod.append(qua_label_divnod)
    subsample_qua_label_divedg.append(qua_label_divedg)
    subsample_qua_feature_divnod.append(qua_feature_divnod)
    subsample_qua_feature_divedg.append(qua_feature_divedg)
    subsample_node_number.append(node_number)
    subsample_edge_number.append(edge_number)
    subsample_runtime.append(total_time)

print(len(idx_list),len(super_node_idx))
print(subsample_meld_correlation)

inf_dict={'subsample_accuracy':subsample_accuracy,'subsample_error':subsample_error,'subsample_qua_label_divnod':subsample_qua_label_divnod,'subsample_qua_label_divedg':subsample_qua_label_divedg,'subsample_qua_feature_divnod':subsample_qua_feature_divnod,'subsample_qua_feature_divedg':subsample_qua_feature_divedg,'subsample_node_number':subsample_node_number,'subsample_edge_number':subsample_edge_number,'subsample_meld_correlation':subsample_meld_correlation,'subsample_coarsening_predict':subsample_coarsening_predict,'subsample_original_predict':subsample_original_predict,'subsample_runtime':subsample_runtime}
get_result=pd.DataFrame(inf_dict)
pick_d=get_result['subsample_node_number']
pick_d=np.array(list(pick_d))

print(np.mean(pick_d, axis=0))
get_result.to_pickle("NK_"+methods+"_1000_new.pkl")