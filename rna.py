import anndata as nd
import numpy as np 
import scanpy as sp
import pandas as pd 
import scanpy as sc
import matplotlib.pyplot as plt

# Function to display the attributes with biological context and clear separation
def display_anndata_attributes_with_context(adata):
    separator = "\n" + "-"*50 + "\n"
    
    # Display shape
    print(f"Shape of the data (cells x genes): {adata.shape}")
    print(separator)
    
    # Display .X attribute
    print(".X (Data Matrix):")
    print("This matrix contains the expression levels of genes across cells.")
    print("Each entry represents the expression level of a gene in a cell.")
    print(f"Data matrix shape: {adata.X.shape}")
    print("Sample data:")
    sample_data = adata.X[:25, :25]  # Show a sample of the first 5 rows and columns
    print(sample_data)
    print(separator)
    
    # Display .obs attribute
    print(".obs (Cell Metadata):")
    print("Metadata for each cell. Each row is a cell, columns include cell-specific info.")
    print(adata.obs.head())
    print(separator)
    
    # Display .obs_names attribute
    print(".obs_names (Cell Identifiers):")
    print("Unique identifiers for each cell.")
    print(adata.obs_names[:5])  # Show a sample of the first 5 cell identifiers
    print(separator)
    
    # Display .var attribute
    print(".var (Gene Metadata):")
    print("Metadata for each gene. Each row is a gene, columns include gene-specific info.")
    print(adata.var.head())
    print(separator)
    
    # Display .var_names attribute
    print(".var_names (Gene Identifiers):")
    print("Unique identifiers for each gene.")
    print(adata.var_names[:5])  # Show a sample of the first 5 gene identifiers
    print(separator)



# Function to visualize number of genes per cell and number of cells per gene with highlighted thresholds
def visualize_gene_and_cell_counts(sc_data, min_genes, min_cells):
    # Calculate the number of genes per cell and number of cells per gene
    sc_data.obs['n_genes'] = sc_data.X.getnnz(axis=1)
    sc_data.var['n_cells'] = sc_data.X.getnnz(axis=0)

    # Create a 1x2 grid of plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot number of genes per cell
    hist_genes, bins_genes = np.histogram(sc_data.obs['n_genes'], bins=50, range=(0, sc_data.obs['n_genes'].max()))
    axes[0].hist(sc_data.obs['n_genes'], bins=bins_genes, color='skyblue', edgecolor='black')
    axes[0].axvline(x=min_genes, color='red', linestyle='--', label=f'Min Genes Threshold: {min_genes}')
    axes[0].set_title('Distribution of Number of Genes per Cell')
    axes[0].set_xlabel('Number of Genes')
    axes[0].set_ylabel('Number of Cells')
    axes[0].legend()
    axes[0].grid(True)

    # Set limits to zoom in on the threshold
    xlim_genes = (0, max(min_genes + 100, sc_data.obs['n_genes'].max()))
    axes[0].set_xlim(xlim_genes)
    axes[0].set_ylim(0, hist_genes.max() + 10)  # Adjust y-limits if needed

    # Plot number of cells per gene
    hist_cells, bins_cells = np.histogram(sc_data.var['n_cells'], bins=50, range=(0, sc_data.var['n_cells'].max()))
    axes[1].hist(sc_data.var['n_cells'], bins=bins_cells, color='lightgreen', edgecolor='black')
    axes[1].axvline(x=min_cells, color='blue', linestyle='--', label=f'Min Cells Threshold: {min_cells}')
    axes[1].set_title('Distribution of Number of Cells per Gene')
    axes[1].set_xlabel('Number of Cells')
    axes[1].set_ylabel('Number of Genes')
    axes[1].legend()
    axes[1].grid(True)

    # Set limits to zoom in on the threshold
    xlim_cells = (0, max(min_cells + 10, sc_data.var['n_cells'].max()))
    axes[1].set_xlim(xlim_cells)
    axes[1].set_ylim(0, hist_cells.max() + 10)  # Adjust y-limits if needed

    plt.tight_layout()
    plt.show()
    
    

# Function to plot the highest expressed genes with improved aesthetics
def plot_highest_genes(sc_data, top_n):
    # Plot top N highest expressed genes
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sc.pl.highest_expr_genes(
        sc_data,
        n_top=top_n,
        ax=ax,
        show=False  # Prevents automatic display; we will manually show it
    )

    # Customize the plot
    ax.set_title(f'Top {top_n} Most Expressed Genes', fontsize=16, weight='bold')
    ax.set_xlabel('Percentage of Total Expression', fontsize=14)
    ax.set_ylabel('Genes', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Optionally adjust the size of the x-axis labels if they overlap
    plt.tight_layout()

    # Display the plot
    plt.show()    

    
def plot_qc_metrics(data, thresholds):
    """
    Annotates ribosomal genes, recalculates QC metrics, and plots QC metrics with thresholds.

    Parameters:
    - data: AnnData object containing single-cell RNA-seq data
    - thresholds: Dictionary containing thresholds for QC metrics
    """
    # Annotate ribosomal genes
    data.var['ribo'] = data.var_names.str.startswith('RPS') | data.var_names.str.startswith('RPL')
    data.var['mt'] = data.var_names.str.startswith('MT-') 
    # Recalculate QC metrics with the new ribosomal annotation
    sc.pp.calculate_qc_metrics(data, qc_vars=['mt', 'ribo'], percent_top=None, log1p=False, inplace=True)

    # Plot additional QC metrics with thresholds
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Plot for number of genes by counts
    sc.pl.violin(data, ['n_genes_by_counts'], jitter=0.4, ax=axs[0, 0], show=False)
    axs[0, 0].axhline(y=thresholds['n_genes_by_counts'], color='red', linestyle='--')
    axs[0, 0].set_title('Number of Genes by Counts')
    axs[0, 0].set_ylabel('Number of Genes')

    # Plot for total counts
    sc.pl.violin(data, ['total_counts'], jitter=0.4, ax=axs[0, 1], show=False)
    axs[0, 1].set_title('Total Counts')
    axs[0, 1].set_ylabel('Total Counts')

    # Plot for percentage of mitochondrial genes
    sc.pl.violin(data, ['pct_counts_mt'], jitter=0.4, ax=axs[1, 0], show=False)
    axs[1, 0].axhline(y=thresholds['pct_counts_mt'], color='red', linestyle='--')
    axs[1, 0].set_title('Percentage of Mitochondrial Genes')
    axs[1, 0].set_ylabel('Percentage')

    # Plot for percentage of ribosomal genes
    sc.pl.violin(data, ['pct_counts_ribo'], jitter=0.4, ax=axs[1, 1], show=False)
    axs[1, 1].set_title('Percentage of Ribosomal Genes')
    axs[1, 1].set_ylabel('Percentage')

    plt.tight_layout()
    plt.show()
    

def extract_top_genes(sc_data, cluster_key='leiden', methods=['t-test', 'wilcoxon', 'logreg']):
    """
    Extract top genes from differential expression analysis and format them into a DataFrame.
    
    Parameters:
    - sc_data: AnnData object containing the single-cell RNA-seq data.
    - cluster_key: The key for the cluster labels in the AnnData object.
    - methods: List of methods to use for differential expression analysis.
    
    Returns:
    - DataFrame containing clusters, top genes from each method, and a final top gene column.
    """
    # Perform differential expression analysis and collect results
    results = {}
    for method in methods:
        sc.tl.rank_genes_groups(sc_data, cluster_key, method=method)
        results[method] = sc_data.uns['rank_genes_groups']
    
    # Extract top genes for each method
    top_genes = {method: {group: results[method]['names'][group][0] for group in results[method]['names'].dtype.names} for method in methods}

    # Combine results into a single DataFrame
    combined_df = pd.DataFrame({
        'Cluster': top_genes[methods[0]].keys(),
        **{f'Top Gene ({method})': [top_genes[method][group] for group in top_genes[methods[0]]] for method in methods}
    })

    # Count occurrences of each gene in all methods
    all_genes = pd.concat([pd.Series(combined_df[f'Top Gene ({method})']) for method in methods])
    gene_counts = all_genes.value_counts()

    # Define the Final Top Gene column
    def determine_final_top_gene(row):
        genes = [row[f'Top Gene ({method})'] for method in methods]
        gene_count = pd.Series(genes).value_counts()
        final_genes = []
        for gene in set(genes):
            if gene_count.get(gene, 0) in [1, 2]:
                final_genes.append(gene)
        if len(final_genes) == 1:
            return final_genes[0]
        elif len(final_genes) == 0:
            return ', '.join(set(genes))
        else:
            return ', '.join(sorted(final_genes))
    
    combined_df['Final Top Gene'] = combined_df.apply(determine_final_top_gene, axis=1)

    # Save to a CSV file
    combined_df.to_csv('combined_top_genes_with_final.csv', index=False)

    return combined_df

