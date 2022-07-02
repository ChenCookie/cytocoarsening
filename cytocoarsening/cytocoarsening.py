import numpy as np
from sklearn.neighbors import kneighbors_graph
import time
import networkx as nx
import sys

def create_KNN_classification(celldataset,k,celllabel):
    A_KNN=kneighbors_graph(celldataset, k,include_self=True, mode="distance")
    A_KNN= np.array(A_KNN.toarray())
    return A_KNN
def get_nonzero_distance_indexv2(node_distance_idx,distance):
    test_idx=list(np.nonzero(distance)[0])
    if node_distance_idx in test_idx:
        test_idx.remove(node_distance_idx)
    return np.array(test_idx)

def get_clumps(nodes_distance, k):
    clumps = np.empty((0,k), int)
    clumps_circle_r = []
    node_index = 0
    for node_distance in range(0,len(nodes_distance),1):
        clump = np.append(node_index, get_nonzero_distance_indexv2(node_distance,nodes_distance[node_distance]))

        if clump.shape[0] != k:
            for i in range(0, k- clump.shape[0]):
                clump = np.append(clump, clump[0])
        clump = np.array([clump])
        clumps = np.append(clumps, clump, axis=0)
        clumps_circle_r = np.append(clumps_circle_r, np.max(nodes_distance[node_distance]))
        node_index+=1
    return [clumps,clumps_circle_r]

def build_original_graph(graph_idx,clumps_array):
    new_graph = nx.Graph()
    for idx in np.sort(graph_idx):new_graph.add_node(idx)
    for sub_clumps in clumps_array:
        for clump_node_index in range(1,len(sub_clumps)):
            if sub_clumps[0]!=sub_clumps[clump_node_index]:
                new_graph.add_edge(sub_clumps[0],sub_clumps[clump_node_index], weight=1)
    return new_graph

def trace_edgeV3(single_clump,graph_nx):
    unique_clump=np.unique(single_clump)
    gett=list(graph_nx.edges(unique_clump))
    edge_track=[(nd1,nd2) for (nd1,nd2) in gett if (nd1 in unique_clump) and  (nd2 in unique_clump)]
    edge_track=np.array([*edge_track])
    edge_track=np.sort(edge_track)
    edge_track=np.unique(edge_track,axis=0)
    # print("edge_track",edge_track)
    return edge_track

def build_graph(graph_idx,graph_edge):
    graph = nx.Graph()
    for idx in np.sort(graph_idx):graph.add_node(idx)
    for edge in graph_edge:graph.add_edge(edge[0], edge[1], weight=1)
    return graph

def get_p(graph,x):
    L=nx.laplacian_matrix(graph)
    return x.T @ L @ x

def rank_product(input_clumps_dist,input_equation_value):
    clumps_dist_sort=np.argsort(np.argsort(np.array(input_clumps_dist)))+1
    equation_value_sort=np.argsort(np.argsort(np.array(input_equation_value)))+1
    combine_rank=(np.log2(clumps_dist_sort)+np.log2(equation_value_sort))*0.5
    return combine_rank

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
        sublabelcount[i] = sublabelcount.get(i, 0) + 1
    sublabelcount=dict(sorted(sublabelcount.items(), key=lambda item: item[1]))
    all_weight_count=np.sum(np.power(np.fromiter(sublabelcount.values(), dtype=int),2))
    for i in range(len(sublabel)):
        weighted[i]+=(sublabelcount[sublabel[i]]/all_weight_count)
    the_min_dix=weighted.index(max(weighted))
    return subgroups[the_min_dix]

def delete_clump_list_data(delete_order,in_clumps_list,in_equation_result,in_clumps_circle_r,in_rank_list):
    del in_clumps_list[delete_order]
    del in_equation_result[delete_order]
    del in_clumps_circle_r[delete_order]
    in_rank_list=np.delete(in_rank_list, delete_order)
    return in_clumps_list,in_equation_result,in_clumps_circle_r,in_rank_list

def compute_dist(edge_node_idx,original_dataC,todo_edge):
    dist_list=[]
    for in_edge in todo_edge:
        idx1=list(edge_node_idx).index(in_edge[0])
        idx2=list(edge_node_idx).index(in_edge[1])
        dist_list.append(np.linalg.norm(original_dataC[idx1] - original_dataC[idx2]))
    # print("compute_dist",edge_node_idx,todo_edge,dist_list)
    return dist_list

def build_nodemap_dict(input_group_dict):
    nodemap_dict=dict()
    for group_nd in input_group_dict.keys():
        for original_node in input_group_dict[group_nd]:
            nodemap_dict[original_node]=group_nd
    return nodemap_dict

def clean_edgesv2(update_edge,group_dict):
    node_map =build_nodemap_dict(group_dict)
    edge_key=0
    update_edge=np.unique(update_edge,axis=0)
    while edge_key<len(update_edge):
        update_edge[edge_key][0]=node_map[update_edge[edge_key][0]]
        update_edge[edge_key][1]=node_map[update_edge[edge_key][1]]
        if update_edge[edge_key][1]==update_edge[edge_key][0]:
            update_edge=np.delete(update_edge,edge_key,axis=0)
        else:
            edge_key+=1
    update_edge=np.sort(update_edge)
    update_edge=np.unique(update_edge,axis=0)
    return update_edge

def trace_node_label_form_dict(groups_dict,labelC,pass_num,k,dataC_idx,acc_str,err_str):
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
    acc_str.append(trace_num/trace_den*100)
    err_str.append(trace_different/trace_den)
    return acc_str,err_str 


def cytocoarsening(input_dataC = None,input_labelC = None,multipass = 10,input_k = 5):
    accuracy_store=list()
    error_store=list()
    qua_label_divnod_store=list()
    qua_feature_divnod_store=list()
    node_store=list()
    edge_store=list()
    runtime_store=list()
    dataC_idx=np.arange(len(input_dataC))
    new_groups=dict()
    keypoint_store=list()

    for singelpsaa in range(multipass):
        count_group=0
        print("multipass",singelpsaa+1)
        total1=time.time()
        if len(dataC_idx)==0 or len(dataC_idx)<input_k:
            break

        A_array = create_KNN_classification(input_dataC[dataC_idx],input_k,input_labelC[dataC_idx])
        [clumps, clumps_circle_r] = get_clumps(A_array, input_k)
        break_threhold=len(clumps)/4
        KNN_graph=build_original_graph(dataC_idx,np.array(dataC_idx)[clumps])

        v2node_group=dict()
        store_edge=np.empty((0,2), int)
        equation_result=[]

        for in_clump in np.array(dataC_idx)[clumps]:
            L_edge=trace_edgeV3(in_clump,KNN_graph)
            store_edge=np.append(store_edge,L_edge,axis=0)
            subgraph=build_graph(in_clump,L_edge)
            qua_result=get_p(subgraph,input_labelC[np.unique(in_clump)])
            equation_result.append(qua_result/len(np.unique(in_clump)))

        mark= np.zeros(len(dataC_idx), dtype=bool)        
        clumps_circle_r=clumps_circle_r.tolist()
        clumps_list = clumps.tolist()
        count_in=0
        
        
        label_threshold=np.percentile(np.sort(equation_result), 26)
        dist_threshold=np.percentile(np.sort(clumps_circle_r), 26)

        rank_pick=30
        count_rank=30
        while len(equation_result)>0:
            if count_rank==rank_pick:
                ranking_list=rank_product(clumps_circle_r,equation_result)
                count_rank=0
            climps_order=np.where(ranking_list == ranking_list.min())[0][0]
            count_rank+=1


            if equation_result[climps_order]>1e10 or clumps_circle_r[climps_order]>1e10:
                break
            if count_in>break_threhold:
                break


            group_list=np.unique(np.array(dataC_idx)[clumps_list[climps_order]])
            i_marked = mark[clumps_list[climps_order]]
            if not any(i_marked):
                mark[clumps_list[climps_order]] = True
                if equation_result[climps_order]>label_threshold or clumps_circle_r[climps_order]>dist_threshold:
                    for group_node in group_list:
                        v2node_group[group_node]=np.array([group_node])
                else:
                    np_node=get_center_point_v2(input_dataC[group_list],group_list,input_labelC[group_list])
                    v2node_group[np_node]=group_list
                    count_group+=1


                clumps_list,equation_result,clumps_circle_r,ranking_list=delete_clump_list_data(climps_order,clumps_list,equation_result,clumps_circle_r,ranking_list)
                count_in+=1


            else:
                remanin_set = np.unique(np.array(clumps_list[climps_order])[~i_marked])
                if len(remanin_set)>0:
                    L_edge=trace_edgeV3(np.array(dataC_idx)[remanin_set],KNN_graph)
                    if len(L_edge)==0 or len(remanin_set)==1:
                        qua_result=sys.maxsize
                        dist_result=sys.maxsize#for dist
                    else:
                        subgraph=build_graph(np.array(dataC_idx)[remanin_set],L_edge)
                        qua_result=get_p(subgraph,input_labelC[np.array(dataC_idx)[remanin_set]])
                        dist_result=np.max(compute_dist(np.array(dataC_idx)[remanin_set],input_dataC[np.array(dataC_idx)[remanin_set]],L_edge))#for dist
        
                    equation_result[climps_order]=qua_result/len(remanin_set)
                    clumps_circle_r[climps_order]=dist_result#for dist
                    clumps_list[climps_order]=np.array(remanin_set)
                    ranking_list[climps_order]=sys.maxsize
                else:
                    clumps_list,equation_result,clumps_circle_r,ranking_list=delete_clump_list_data(climps_order,clumps_list,equation_result,clumps_circle_r,ranking_list)


        if len(clumps_list)>0:
            for single_node in np.array(dataC_idx)[~mark]:
                v2node_group[single_node]=np.array([single_node])

        for i in list(v2node_group.values()):
            build_group=i
            supernd_intersection=list(set(build_group).intersection(set(new_groups.keys())))
            if len(supernd_intersection)>0:
                for spnd in supernd_intersection:
                    build_group=list(set(build_group).union(set(new_groups[spnd])))
                    del new_groups[spnd]
            np_node=get_center_point_v2(input_dataC[build_group],build_group,input_labelC[build_group])
            new_groups[np_node]=build_group

        store_cleanedge=clean_edgesv2(store_edge,new_groups)
        dataC_idx=list(new_groups.keys())
        dataC_idx.sort()

        node_store.append(len(dataC_idx))
        subgraph=build_graph(dataC_idx,store_cleanedge)
        qua_result=get_p(subgraph,input_labelC[dataC_idx])
        qua_label_divnod_store.append(qua_result/len(dataC_idx))
        edge_store.append(len(store_cleanedge))

        euqation_result=[]
        for i in range(0,input_dataC.shape[1],1):
            euqation_result.append(get_p(subgraph,input_dataC[dataC_idx][:,i]))
        qua_feature_divnod_store.append((sum(euqation_result) / len(euqation_result))/len(dataC_idx))

        accuracy_store,error_store=trace_node_label_form_dict(new_groups,input_labelC,0,0,dataC_idx,accuracy_store,error_store)
        total2=time.time()
        runtime_store.append(total2-total1)
        keypoint_store.append(dataC_idx)
        if(count_group==0):
            break

    information_dict={'accuracy':accuracy_store,'error':error_store,'qua_label_divnod':qua_label_divnod_store,\
    'qua_feature_divnod':qua_feature_divnod_store,'node_number':node_store,'edge_number':edge_store,'runtime':runtime_store,\
    'keypoint':keypoint_store}
    return new_groups,store_cleanedge,information_dict


