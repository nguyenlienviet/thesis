import os, subprocess
import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests
from igraph import Graph, plot
from collections import defaultdict
import itertools
from scipy.stats import spearmanr


def run_sparcc(df, outdir, force=False):
    raw_path = os.path.join(outdir, 'raw.tsv')
    cor_path = os.path.join(outdir, 'cor.tsv')
    cov_path = os.path.join(outdir, 'cov.tsv')
    bs_counts_path = os.path.join(outdir, 'bootstrap_counts')
    bs_cor_path = os.path.join(outdir, 'bootstrap_correlation')
    pval_path = os.path.join(outdir, 'pvalues.tsv')

    if not os.path.exists(raw_path) or force:
        df.rename_axis('#OTU ID').to_csv(raw_path, sep='\t')

    if not os.path.exists(cor_path) or not os.path.exists(cov_path) or force:
        subprocess.run(['fastspar-0.0.10_linux/fastspar',
                        '-c', raw_path,
                        '-r', cor_path,
                        '-a', cov_path],
                       check=True)
    
    if not os.path.exists(bs_counts_path) or force:
        os.makedirs(bs_counts_path, exist_ok=True)
        subprocess.run(['fastspar-0.0.10_linux/fastspar_bootstrap',
                        '--otu_table', raw_path,
                        '--number', '1000',
                        '--prefix', os.path.join(bs_counts_path, 'bs')],
                       check=True)

    if not os.path.exists(bs_cor_path) or force:
        os.makedirs(bs_cor_path, exist_ok=True)
        # subprocess.run(['parallel', 'fastspar-0.0.10_linux/fastspar',
        #                 '--otu_table', '{}',
        #                 '--correlation', os.path.join(bs_cor_path, 'cor_{/}'),
        #                 '--covariance', os.path.join(bs_cor_path, 'cov_{/}'),
        #                 '-i', '5', '-y', ':::', os.path.join(bs_counts_path, '*')],
        #               check=True, shell=True)
        os.system(' '.join(['parallel', 'fastspar-0.0.10_linux/fastspar',
                        '--otu_table', '{}',
                        '--correlation', os.path.join(bs_cor_path, 'cor_{/}'),
                        '--covariance', os.path.join(bs_cor_path, 'cov_{/}'),
                        '-i', '5', '-y', ':::', os.path.join(bs_counts_path, '*')]))

    if not os.path.exists(pval_path) or force:
        subprocess.run(['fastspar-0.0.10_linux/fastspar_pvalues',
                        '--otu_table', raw_path,
                        '--correlation', cor_path,
                        '--prefix', os.path.join(bs_cor_path, 'cor_bs_'),
                        '--permutations', '1000',
                        '--outfile', pval_path],
                       check=True)

    df_cor = pd.read_csv(cor_path, sep='\t', index_col='#OTU ID')
    # df_cov = pd.read_csv(cov_path, sep='\t', index_col='#OTU ID')
    df_pval = pd.read_csv(pval_path, sep='\t', index_col='#OTU ID')
    return df_cor, df_pval

def correct_pvals(df_pval, alpha=.05):
    flatten = flatten_square_df(df_pval, k=1)
    reject, pvals_corrected, _, _ = multipletests(flatten, alpha=alpha,
                                                  method='fdr_bh')
    return flatten[reject].max() if sum(reject) > 0 else 0

def correct_alpha(pvals, alpha=.05):
    reject, pvals_corrected, _, _ = multipletests(pvals, alpha=alpha,
                                                  method='fdr_bh')
    return pvals[reject].max() if sum(reject) > 0 else 0

def flatten_square_df(df, k=1):
    upper_triangle = df.where(np.triu(np.ones(df.shape), k=k).astype(np.bool))
    melted =  upper_triangle.stack()
    return melted

def subset_by_season(df, meta, season):
    samples = meta['Sample_Names'][meta['Season'] == season]
    df = df[samples]
    df = df[df.sum(axis=1) > 0]
    return df

def filter_by_abundance(df, min_abundance):
    df_relative = df / df.sum()
    return df[(df_relative >= min_abundance).any(axis=1)]

def filter_by_prevalence(df, min_prevalance):
    return df[(df > 0).sum(axis=1) >= min_prevalance]

def get_supernode_to_members(member_to_supernode):
    supernode_to_members = defaultdict(set)
    for member, supernode in member_to_supernode.items():
        supernode_to_members[supernode].add(member)
    return supernode_to_members

def find_valid_grouping(G, supernode_to_members, level):
    togroup = set()
    for supernode, members in supernode_to_members.items():
        if len(supernode) < level:
            continue
        group = True
        for m1, m2 in itertools.combinations(members, 2):
            if m1 in G and m2 in G:
                try:
                    if G[m1][m2]['weight'] < 0:
                        group = False
                        break
                except KeyError:
                    pass
                for neighbor, edge1 in G[m1].items():
                    try:
                        edge2 = G[m2][neighbor]
                        if np.sign(edge1['weight']) != np.sign(edge2['weight']):
                            group = False
                            break
                    except KeyError:
                        pass
                if not group:
                    break
        if group:
            togroup.add(supernode)
    return togroup

def merge_nodes(G, nodes, new_node):
    if len(nodes) == 1:
        return
    G.add_node(new_node)
    
    for node in nodes:
        for neighbor, edge in G[node].items():
            if neighbor in nodes or neighbor in G[new_node]:
                continue
            weights = []
            for node in nodes:
                if neighbor in G[node]:
                    weights.append(G[node][neighbor]['weight'])
            G.add_edge(new_node, neighbor, weight=np.mean(weights))

    # G.nodes[new_node]['abundance'] = np.mean([G.nodes[n]['abundance'] for n in nodes])
    # G.nodes[new_node]['taxonomy'] = G.nodes[nodes[0]]['taxonomy']
    
    G.nodes[new_node]['OTUs'] = 0
    for n in nodes:
        # G.nodes[new_node]['taxonomy'] = G.nodes[n]['taxonomy']
        G.nodes[new_node]['OTUs'] += G.nodes[n]['OTUs']
        G.remove_node(n)

def get_member_to_supernode(G, level):
    member_to_supernode = dict()
    for node in G:
        member_to_supernode[node] = node[:level]
    return member_to_supernode

def merge_nodes_to_level(G, from_level, to_level):
    print(f'Original #nodes: {len(G)}')
    for level in range (from_level, to_level-1, -1):
        member_to_supernode = get_member_to_supernode(G, level)
        supernode_to_members = get_supernode_to_members(member_to_supernode)
        togroup = find_valid_grouping(G, supernode_to_members, level)
        for supernode in togroup:
            merge_nodes(G, supernode_to_members[supernode], supernode)
        print(f'Level {level} #nodes: {len(G)}')

def add_node_attrs(G, level, df_filtered_relative, OTU_to_tax):
    for node, attrs in G.nodes.items():
        attrs['taxonomy'] = '; '.join(node[:min(len(node), level)])
        if node[-1].startswith('denovo'):
            attrs['abundance'] = df_filtered_relative.loc[node[-1]].mean()
        else:
            OTUs = [OTU for OTU, tax in OTU_to_tax.items() if len(tax) >= len(node) and tax[:len(node)] == node]
            attrs['abundance'] = df_filtered_relative.loc[OTUs].mean().mean()

def add_edge_attrs(G):
    for edge, attrs in G.edges.items():
        attrs['absweight'] = abs(attrs['weight'])
        attrs['color'] = 'green' if attrs['weight'] > 0 else 'red'

def calc_spearman_cor(df_OTU, env_var):
    nonnan_samples = env_var.index[env_var.notna()]
    df_OTU = df_OTU.loc[:, nonnan_samples]
    env_var = env_var[nonnan_samples]
    
    env_cor_pval = df_OTU.apply(lambda row: spearmanr(row, env_var), axis=1, result_type='expand')
    env_cor = env_cor_pval[0]
    env_pval = env_cor_pval[1]
    threshold = correct_alpha(env_pval)
    env_cor.loc[env_cor[env_pval > threshold].index] = 0
    env_cor.name = env_var.name
    return env_cor

def add_env_node(G, env_cor, OTU_to_tax):
    node_name = env_cor.name
    G.add_node(node_name)
    for n in list(G):
        try:
            n = tuple(n.split('; '))
        except AttributeError:
            pass
        if n[-1].startswith('denovo'):
            weight = env_cor[n[-1]]
        else:
            weights = []
            for OTU, tax in OTU_to_tax.items():
                if len(tax) >= len(n) and tax[:len(n)] == n:
                    weights.append(env_cor[OTU])
            weight = np.mean(weights)
        if weight != 0 and not np.isnan(weight):
            G.add_edge(node_name, '; '.join(n), weight=weight, color='red' if weight < 0 else 'green', absweight=abs(weight))
            pass

def graph_stats(g):
    print(f'# edges: {len(g.es)}')
    print(f'# positive edges: {(np.array(g.es["weight"]) > 0).sum()}')
    print(f'# negative edges: {(np.array(g.es["weight"]) < 0).sum()}')
    print(f'max degree: {np.max(g.degree())}')
    print(f'min degree: {np.min(g.degree())}')
    print(f'median degree: {np.median(g.degree())}')
    print(f'transitivity: {g.transitivity_undirected()}')