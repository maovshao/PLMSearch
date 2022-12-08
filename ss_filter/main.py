"""
Created on 2021/12/28
@author liuwei
"""
from ss_filter_util.statistics_util import get_input_output, cluster_statistics, ss_mat_statistics, esm_similarity_statistics, ss_predictor_statistics, get_miss_wrong_statistics, venn_graph3, pair_list_statistics, scop_roc, tmscore_aupr, tmscore_precision_recall

if __name__ == '__main__':
    # Pfam Compare(scope)
    todo_dir_list = ["./ss_filter_data/scope_test/prefilter_result/"]
    ss_mat_path = "./ss_filter_data/scope_test/ss_mat.npy"
    query_protein_list_path = "./ss_filter_data/scope_test/protein_list.txt"
    target_protein_list_path = "./ss_filter_data/scope_test/protein_list.txt"
    todo_name_list = [
        'pfamfamily',
        'pfamclan'
        ]
    todo_file_list, todo_fig_list = get_input_output(todo_dir_list, todo_name_list)
    method_list = ['PfamFamily', 'PfamClan']
    top_list = ['all', 'all']
    ss_mat_statistics(ss_mat_path, query_protein_list_path, target_protein_list_path, todo_file_list, todo_fig_list, "./scientist_figures/pfam_scope_compare.png", method_list, top_list)

    # Pfam Compare(swissprot_to_swissprot)
    todo_dir_list = ["./ss_filter_data/swissprot_to_swissprot/prefilter_result/"]
    ss_mat_path = "./ss_filter_data/swissprot_to_swissprot/ss_mat.npy"
    query_protein_list_path = "./ss_filter_data/swissprot_to_swissprot/query_protein_list.txt"
    target_protein_list_path = "./ss_filter_data/swissprot_to_swissprot/target_protein_list.txt"
    todo_name_list = [
        'pfamfamily',
        'pfamclan'
        ]
    todo_file_list, todo_fig_list = get_input_output(todo_dir_list, todo_name_list)
    method_list = ['PfamFamily', 'PfamClan']
    top_list = ['all', 'all']
    ss_mat_statistics(ss_mat_path, query_protein_list_path, target_protein_list_path, todo_file_list, todo_fig_list, "./scientist_figures/pfam_swissprot_to_swissprot_compare.png", method_list, top_list)

    # Pfam Compare(scope_to_swissprot)
    todo_dir_list = ["./ss_filter_data/scope_to_swissprot/prefilter_result/"]
    ss_mat_path = "./ss_filter_data/scope_to_swissprot/ss_mat.npy"
    query_protein_list_path = "./ss_filter_data/scope_to_swissprot/query_protein_list.txt"
    target_protein_list_path = "./ss_filter_data/scope_to_swissprot/target_protein_list.txt"
    todo_name_list = [
        'pfamfamily',
        'pfamclan'
        ]
    todo_file_list, todo_fig_list = get_input_output(todo_dir_list, todo_name_list)
    method_list = ['PfamFamily', 'PfamClan']
    top_list = ['all', 'all']
    ss_mat_statistics(ss_mat_path, query_protein_list_path, target_protein_list_path, todo_file_list, todo_fig_list, "./scientist_figures/pfam_scope_to_swissprot_compare.png", method_list, top_list)

    # cluster statistics(scope test)
    pfam_result = "./ss_filter_data/scope_test/pfam_result.json"
    clan_file = './ss_filter_data/Pfam_db/Pfam-A.clans.tsv'
    result_path = './scientist_figures/scope_test/'
    cluster_statistics(pfam_result, clan_file, result_path)

    # cluster statistics(swissprot)
    pfam_result = "./ss_filter_data/swissprot_to_swissprot/target_pfam_result.json"
    clan_file = './ss_filter_data/Pfam_db/Pfam-A.clans.tsv'
    result_path = './scientist_figures/swissprot_to_swissprot/'
    cluster_statistics(pfam_result, clan_file, result_path)

    # All-versus-all search on SCOPe40-test
    alnresult_dir = "./ss_filter_data/scope_test/rocx/"
    methods_filename_list = ["mmseqs2.rocx", "3dblastsw.rocx", "clesw.rocx", "foldseek.rocx", "cealn.rocx", "dalialn.rocx", "tmaln.rocx", "mse.rocx", "cos.rocx", "ss_sort.rocx", "ss_sort_cos.rocx", "ss_filter.rocx"]
    roc_plot_name = "./scientist_figures/scop_roc"
    methods_name_list = ["MMseqs2", "3D-BLAST-SW", "CLE-SW", "Foldseek", "CE", "Dali", "TM-align", "Euclidean", "COS", "SS-sort", "SS-sort(COS)", "SS-filter"]
    scop_roc(alnresult_dir, methods_filename_list, roc_plot_name, methods_name_list)

    # Evaluation based on TM-score benchmark(Metrics for evaluating different search methods) on SCOPe40-test
    todo_dir_list = ["./ss_filter_data/scope_test/prefilter_result/"]
    ss_mat_path = "./ss_filter_data/scope_test/ss_mat.npy"
    protein_list_path = "./ss_filter_data/scope_test/protein_list.txt"
    todo_name_list = [
        "mmseqs2", 
        "3dblastsw", 
        "clesw",
        "foldseek",
        "cealn", 
        "dalialn", 
        "mse", 
        "cos", 
        "ss_sort", 
        "ss_sort_cos", 
        "ss_filter"
    ]
    todo_file_list, _ = get_input_output(todo_dir_list, todo_name_list)
    method_list = ["MMseqs2", "3D-BLAST-SW", "CLE-SW", "Foldseek", "CE", "Dali", "Euclidean", "COS", "SS-sort", "SS-sort(COS)", "SS-filter"]
    tmscore_aupr(ss_mat_path, protein_list_path, protein_list_path, todo_file_list, "./scientist_figures/tmscore_aupr_scope", method_list)
    k = 10
    tmscore_precision_recall(ss_mat_path, protein_list_path, protein_list_path, todo_file_list, method_list, k=k)

    # Evaluation based on TM-score benchmark(Metrics for evaluating different search methods) on Swiss-Prot to Swiss-Prot
    todo_dir_list = ["./ss_filter_data/swissprot_to_swissprot/prefilter_result/"]
    ss_mat_path = "./ss_filter_data/swissprot_to_swissprot/ss_mat.npy"
    query_protein_list_path = "./ss_filter_data/swissprot_to_swissprot/query_protein_list.txt"
    target_protein_list_path = "./ss_filter_data/swissprot_to_swissprot/target_protein_list.txt"
    todo_name_list = [
        "mmseqs2", 
        "foldseek", 
        "mse", 
        "cos", 
        "ss_sort",
        "ss_sort_cos",
        "ss_filter"
    ]
    todo_file_list, _ = get_input_output(todo_dir_list, todo_name_list)
    method_list = ["MMseqs2", "Foldseek", "Euclidean", "COS", "SS-sort", "SS-sort(COS)", "SS-filter"]
    tmscore_aupr(ss_mat_path, query_protein_list_path, target_protein_list_path, todo_file_list, "./scientist_figures/tmscore_aupr_swissprot_to_swissprot", method_list)
    k = 100
    tmscore_precision_recall(ss_mat_path, query_protein_list_path, target_protein_list_path, todo_file_list, method_list, k=k)

    #Evaluation based on TM-score benchmark(Metrics for evaluating different search methods) on SCOPe40 to Swiss-Prot
    todo_dir_list = ["./ss_filter_data/scope_to_swissprot/prefilter_result/"]
    ss_mat_path = "./ss_filter_data/scope_to_swissprot/ss_mat.npy"
    query_protein_list_path = "./ss_filter_data/scope_to_swissprot/query_protein_list.txt"
    target_protein_list_path = "./ss_filter_data/scope_to_swissprot/target_protein_list.txt"
    todo_name_list = [
        "mmseqs2", 
        "foldseek", 
        "mse", 
        "cos", 
        "ss_sort",
        "ss_sort_cos",
        "ss_filter"
    ]
    todo_file_list, _ = get_input_output(todo_dir_list, todo_name_list)
    method_list = ["MMseqs2", "Foldseek", "Euclidean", "COS", "SS-sort", "SS-sort(COS)", "SS-filter"]
    tmscore_aupr(ss_mat_path, query_protein_list_path, target_protein_list_path, todo_file_list, "./scientist_figures/tmscore_aupr_scope_to_swissprot", method_list)
    k = 100
    tmscore_precision_recall(ss_mat_path, query_protein_list_path, target_protein_list_path, todo_file_list, method_list, k=k)

    # ablation1, scope roc
    alnresult_dir = "./ss_filter_data/scope_test/rocx/"
    methods_filename_list = ["tmaln.rocx", "tmaln_filter.rocx", "mse.rocx", "cos.rocx", "ss_sort.rocx", "ss_sort_cos.rocx", "ss_filter.rocx"]
    roc_plot_name = "./scientist_figures/scop_roc_ablation"
    methods_name_list = ["TMalign", "TMalign-filter", "Euclidean", "COS", "SS-sort", "SS-sort(COS)", "SS-filter"]
    scop_roc(alnresult_dir, methods_filename_list, roc_plot_name, methods_name_list)

    # ablation2, Evaluation result based on TM-score benchmark(Metrics for evaluating different pre-filter methods)
    todo_dir_list = ["./ss_filter_data/swissprot_to_swissprot/prefilter_result/"]
    ss_mat_path = "./ss_filter_data/swissprot_to_swissprot/ss_mat.npy"
    query_protein_list_path = "./ss_filter_data/swissprot_to_swissprot/query_protein_list.txt"
    target_protein_list_path = "./ss_filter_data/swissprot_to_swissprot/target_protein_list.txt"
    todo_name_list = [
        'pfamclan',
        'mse',
        'cos',
        'ss_sort',
        'ss_sort_cos',
        'ss_filter'
        ]
    todo_file_list, todo_fig_list = get_input_output(todo_dir_list, todo_name_list)
    method_list = ['PfamClan', 'Euclidean', 'COS', 'SS-sort', 'SS-sort(COS)', 'SS-filter']
    top_list = ['all', 10000, 10000, 10000, 10000, 10000]
    ss_mat_statistics(ss_mat_path, query_protein_list_path, target_protein_list_path, todo_file_list, todo_fig_list, "./scientist_figures/swissprot_to_swissprot/ablation/distribution.png", method_list, top_list)

    #miss fault statistics(ablation)
    todo_dir_list = ["./ss_filter_data/swissprot_to_swissprot/prefilter_result/"]
    ss_mat_path = "./ss_filter_data/swissprot_to_swissprot/ss_mat.npy"
    query_protein_list_path = "./ss_filter_data/swissprot_to_swissprot/query_protein_list.txt"
    target_protein_list_path = "./ss_filter_data/swissprot_to_swissprot/target_protein_list.txt"
    result_path = './scientist_figures/swissprot_to_swissprot/ablation/'
    todo_name_list = [
        'pfamclan',
        'ss_sort',
        'ss_filter'
        ]
    todo_file_list, todo_fig_list = get_input_output(todo_dir_list, todo_name_list)
    top_list = ['all', 10000, 10000]
    get_miss_wrong_statistics(ss_mat_path, query_protein_list_path, target_protein_list_path, todo_file_list, top_list, result_path)

    #venn_graph3(ablation)
    pfamclan_missed_filename = './scientist_figures/swissprot_to_swissprot/ablation/miss_pfamclan.txt'
    ss_sort_missed_filename = './scientist_figures/swissprot_to_swissprot/ablation/miss_ss_sort.txt'
    ss_filter_missed_filename = './scientist_figures/swissprot_to_swissprot/ablation/miss_ss_filter.txt'
    venn_graph_name = './scientist_figures/swissprot_to_swissprot/ablation/venn_graph_miss.png'
    venn_graph3([pfamclan_missed_filename, ss_sort_missed_filename, ss_filter_missed_filename], ('PfamClan->SS-filter', 'SS-sort', ''), venn_graph_name)

    pfamclan_wrong_filename = './scientist_figures/swissprot_to_swissprot/ablation/wrong_pfamclan.txt'
    ss_sort_wrong_filename = './scientist_figures/swissprot_to_swissprot/ablation/wrong_ss_sort.txt'
    ss_filter_wrong_filename = './scientist_figures/swissprot_to_swissprot/ablation/wrong_ss_filter.txt'
    venn_graph_name = './scientist_figures/swissprot_to_swissprot/ablation/venn_graph_wrong.png'
    venn_graph3([pfamclan_wrong_filename, ss_sort_wrong_filename, ss_filter_wrong_filename], ('PfamClan', 'SS-sort', 'SS-filter'), venn_graph_name)

    # ss_esm_similarity statistics(swissprot_to_swissprot)
    query_esm_filename = './ss_filter_data/swissprot_to_swissprot/query_mean_esm_result.pkl'
    target_esm_filename = './ss_filter_data/swissprot_to_swissprot/target_mean_esm_result.pkl'
    #if cpu only
    #query_esm_filename = './ss_filter_data/swissprot_to_swissprot/query_mean_esm_result_cpu.pkl'
    #target_esm_filename = './ss_filter_data/swissprot_to_swissprot/target_mean_esm_result_cpu.pkl'
    query_protein_list_path = "./ss_filter_data/swissprot_to_swissprot/query_protein_list.txt"
    target_protein_list_path = "./ss_filter_data/swissprot_to_swissprot/target_protein_list.txt"
    ss_mat_path = "./ss_filter_data/swissprot_to_swissprot/ss_mat.npy"

    fig_name = f'./scientist_figures/esm_similarity_mse_statistics_swissprot_to_swissprot.png'
    esm_similarity_statistics(query_esm_filename, target_esm_filename, query_protein_list_path, target_protein_list_path, ss_mat_path, fig_name, mode='mse')

    fig_name = f'./scientist_figures/esm_similarity_cos_statistics_swissprot_to_swissprot.png'
    esm_similarity_statistics(query_esm_filename, target_esm_filename, query_protein_list_path, target_protein_list_path, ss_mat_path, fig_name, mode='cos')

    # ss_predictor_score statistics(swissprot_to_swissprot)
    save_model_path = './ss_filter_data/esm_ss_predict/model_scop_tri.sav'
    device_id = [0]
    # if cpu only
    # device_id = [] 
    fig_name = f'./scientist_figures/ss_predictor_statistics_swissprot_to_swissprot.png'
    ss_predictor_statistics(query_esm_filename, target_esm_filename, query_protein_list_path, target_protein_list_path, ss_mat_path, save_model_path, device_id, fig_name, cos = False)

    save_model_path = './ss_filter_data/esm_ss_predict/model_scop_tri.sav'
    device_id = [0]
    # if cpu only
    # device_id = [] 
    fig_name = f'./scientist_figures/ss_predictor_cos_statistics_swissprot_to_swissprot.png'
    ss_predictor_statistics(query_esm_filename, target_esm_filename, query_protein_list_path, target_protein_list_path, ss_mat_path, save_model_path, device_id, fig_name, cos = True)

    # miss fault statistics(swissprot_to_swissprot, get & miss)
    todo_dir_list = ["./ss_filter_data/swissprot_to_swissprot/prefilter_result/"]
    ss_mat_path = "./ss_filter_data/swissprot_to_swissprot/ss_mat.npy"
    query_protein_list_path = "./ss_filter_data/swissprot_to_swissprot/query_protein_list.txt"
    target_protein_list_path = "./ss_filter_data/swissprot_to_swissprot/target_protein_list.txt"
    result_path = './scientist_figures/swissprot_to_swissprot/get_miss/'
    todo_name_list = [
        'mmseqs2',
        'foldseek',
        'ss_sort_cos',
        'ss_filter'
        ]
    todo_file_list, todo_fig_list = get_input_output(todo_dir_list, todo_name_list)
    top_list = ['all', 5506, 5506, 5506]
    get_miss_wrong_statistics(ss_mat_path, query_protein_list_path, target_protein_list_path, todo_file_list, top_list, result_path)

    #pair list statistics
    pair_list_filename_list = []
    pair_list_filename_list.append("./scientist_figures/swissprot_to_swissprot/get_miss/get_mmseqs2.txt")
    pair_list_filename_list.append("./scientist_figures/swissprot_to_swissprot/get_miss/miss_mmseqs2.txt")
    pair_list_filename_list.append("./scientist_figures/swissprot_to_swissprot/get_miss/get_foldseek.txt")
    pair_list_filename_list.append("./scientist_figures/swissprot_to_swissprot/get_miss/miss_foldseek.txt")
    pair_list_filename_list.append("./scientist_figures/swissprot_to_swissprot/get_miss/get_ss_sort_cos.txt")
    pair_list_filename_list.append("./scientist_figures/swissprot_to_swissprot/get_miss/miss_ss_sort_cos.txt")
    pair_list_filename_list.append("./scientist_figures/swissprot_to_swissprot/get_miss/get_ss_filter.txt")
    pair_list_filename_list.append("./scientist_figures/swissprot_to_swissprot/get_miss/miss_ss_filter.txt")
    for pair_list_filename in pair_list_filename_list:
        query_fasta_filename = "./ss_filter_data/swissprot_to_swissprot/query_protein.fasta"
        target_fasta_filename = "./ss_filter_data/swissprot_to_swissprot/target_protein.fasta"
        ss_mat_path = "./ss_filter_data/swissprot_to_swissprot/ss_mat.npy"
        query_protein_list_path = "./ss_filter_data/swissprot_to_swissprot/query_protein_list.txt"
        target_protein_list_path = "./ss_filter_data/swissprot_to_swissprot/target_protein_list.txt"
        pair_list_statistics(pair_list_filename, query_fasta_filename, target_fasta_filename, query_protein_list_path, target_protein_list_path, ss_mat_path)