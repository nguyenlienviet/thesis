import os, subprocess
import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests
from igraph import Graph, plot

def run_sparcc(df, outdir, force=False):
    os.makedirs(outdir, exist_ok=True)

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

def correct_pvals(df_pval):
    flatten = flatten_square_df(df_pval, k=1)
    reject, pvals_corrected, _, _ = multipletests(flatten, alpha=.99,
                                                  method='fdr_bh')
    return reject, pvals_corrected

def flatten_square_df(df, k=1):
    upper_triangle = df.where(np.triu(np.ones(df.shape), k=k).astype(np.bool))
    melted =  upper_triangle.stack()
    return melted

def build_graph(df_cor, df_pvalue):
    filt = (df_pvalue.values < 0.05) & (np.abs(df_cor.values) > 0)
    adjmatrix = np.where(filt, df_cor.values, 0)
    g = Graph.Weighted_Adjacency(adjmatrix, mode='UNDIRECTED')
    g.vs['label'] = df_cor.index
    g.vs.select(_degree=0).delete()
    return g

def graph_stats(g):
    print(f'# edges: {len(g.es)}')
    print(f'# positive edges: {(np.array(g.es["weight"]) > 0).sum()}')
    print(f'# negative edges: {(np.array(g.es["weight"]) < 0).sum()}')
    print(f'max degree: {np.max(g.degree())}')
    print(f'min degree: {np.min(g.degree())}')
    print(f'median degree: {np.median(g.degree())}')
    print(f'transitivity: {g.transitivity_undirected()}')

def plot_graph(g):
    g.es.select(weight_lt=0)['color'] = 'red'
    g.es.select(weight_gt=0)['color'] = 'green'
    g.vs.select(cluster_eq=0.0)['color'] = 'gray'
    g.vs.select(cluster_eq=1.0)['color'] = 'black'
    labels = g.vs['label']
    g.vs['label'] = None
    p = plot(g, layout=g.layout('kk'))
    return p, labels