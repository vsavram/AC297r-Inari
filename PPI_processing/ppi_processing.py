import pandas as pd
import pickle
import numpy as np
import re

def get_mapping(fpath):
  df = pd.read_csv(fpath, delimiter='\t', header=None)
  mapping = {}
  for i,row_data in df.iterrows():
    # genes_to_keep = [bool(re.match(r'Zm00001d|B73v3', str(x))) for x in row_data]
    output_genes = [x for x in row_data if bool(re.match(r'^Zm00001d', str(x)))]
    # output_genes = tuple(output_genes)
    if not output_genes:
      continue

    for x in row_data:
      if bool(re.match(r'^B73v3_', str(x))):
        gene = x.split('_')[1]
        mapping[gene] = output_genes
  return mapping

# Replaces PPI gene names with Zm00001d version gene names
def process_ppi(fpath, mapping):
  df = pd.read_csv(fpath, delimiter='\t', header=None)
  # remove protein suffix
  df = df.apply(lambda s: s.str.split('_', expand=True)[0])
  # replace with Zm00001d version gene names
  df = df.applymap(lambda s: mapping.get(s, None))
  # explode the many-to-many mappings
  df = df.explode(0).explode(1)
  df = df.dropna()
  return df.T

def process_expr(fpath):
  expr = pd.read_csv(fpath, delimiter='\t')
  # Only leaves genes in the columns
  expr = expr.drop(columns=['growth_condition', 'Cultivar', 'Developmental_stage', 'organism_part', 'Age'])
  # Sets 'Run' as the index and drop the row
  expr = expr.set_index('Run', drop=True)
  # Transpose the expression matrix as per the Graph Feature Autoencoder model
  return expr.T

def convert_gene_names_to_int(ppi_df, expr_df):
  overlapping_genes = np.unique(ppi_df)
  # In expressions, only keep the genes that exist in the PPI
  expr_df = expr_df.loc[overlapping_genes]
  num_genes = expr_df.shape[0]
  # Dict that maps gene names as keys to corresponding index int as values
  genes_dict = dict(zip(expr_df.index.to_numpy(), range(num_genes)))
  # In PPI, replace gene names with ints representing the genes
  ppi_df = ppi_df.applymap(lambda g: genes_dict.get(g, None))
  # Switch key and value in genes dict for easier lookup from int to name
  genes_idx_name_dict = {idx:name for name,idx in genes_dict.items()}
  return (ppi_df.to_numpy(), expr_df.to_numpy(), genes_idx_name_dict)

def save_npy(df, fpath):
  with open(fpath, 'wb') as f:
    np.save(f, df)

def save_pickle(obj, fpath):
  with open(fpath, 'wb') as f:
    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
  mapping = get_mapping('MaizeGDB_pangene.tsv')

  # Note that PPI and expression processing might be particular to these datasets
  ppi_df = process_ppi('original/HighPPIs.txt', mapping)
  expr_df = process_expr('original/combined_expressions.txt')

  ppi_df, expr_df, genes_idx_name_dict = convert_gene_names_to_int(ppi_df, expr_df)

  save_npy(ppi_df, 'processed/maize_high_ppi.npy')
  save_npy(expr_df, 'processed/maize_expressions.npy')
  save_pickle(genes_idx_name_dict, 'processed/genes_dict_high_ppi.pickle')
