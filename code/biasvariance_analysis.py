import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

def compute_bootstrap_coefficients(gene_expression, drug_response, n_bootstraps=100, n_alphas=50):
    """
    Compute coefficient profiles using bootstrap resampling
    
    Parameters:
    gene_expression: DataFrame with gene expression data
    drug_response: DataFrame with drug response data  
    n_bootstraps: Number of bootstrap iterations
    n_alphas: Number of alpha values to test
    
    Returns:
    Dictionary with coefficient profiles and analysis results
    """
    # Set up the alpha grid
    alphas = np.logspace(-3, 2, n_alphas)
    
    # Prepare storage for coefficients
    coefficient_profiles = {gene: np.zeros((n_bootstraps, len(alphas))) 
                           for gene in gene_expression.columns}
    
    # Bootstrap loop
    print("Computing bootstrap coefficients...")
    for i in range(n_bootstraps):
        if i % 10 == 0:  # Progress update
            print(f"Completed {i}/{n_bootstraps} bootstrap iterations")
        
        # Bootstrap sample
        X_boot, y_boot = resample(gene_expression, drug_response.values.ravel(), random_state=i)
        
        # Standardize features
        scaler = StandardScaler()
        X_boot_scaled = scaler.fit_transform(X_boot)
        
        # Fit Lasso for each alpha
        for j, alpha in enumerate(alphas):
            lasso = Lasso(alpha=alpha, max_iter=50000, random_state=42)
            lasso.fit(X_boot_scaled, y_boot)
            
            # Store coefficients for each gene
            for k, gene in enumerate(gene_expression.columns):
                coefficient_profiles[gene][i, j] = lasso.coef_[k]
    
    # Compute additional analysis metrics
    print("Computing analysis metrics...")
    non_zero_counts = np.zeros(len(alphas))
    avg_variance = np.zeros(len(alphas))
    
    for j, alpha in enumerate(alphas):
        # Count non-zero coefficients
        non_zero_count = 0
        for gene in gene_expression.columns:
            mean_coef = np.mean(coefficient_profiles[gene][:, j])
            if abs(mean_coef) > 1e-6:
                non_zero_count += 1
        non_zero_counts[j] = non_zero_count
        
        # Compute average variance
        variances = []
        for gene in gene_expression.columns:
            variances.append(np.var(coefficient_profiles[gene][:, j]))
        avg_variance[j] = np.mean(variances)
    
    # Select representative genes for plotting
    selected_genes = []
    for gene in gene_expression.columns:
        max_coef = np.max(np.abs(np.mean(coefficient_profiles[gene], axis=0)))
        if max_coef > 0.1:
            selected_genes.append(gene)
        if len(selected_genes) >= 12:
            break
    
    # Return all computed data
    results = {
        'coefficient_profiles': coefficient_profiles,
        'alphas': alphas,
        'non_zero_counts': non_zero_counts,
        'avg_variance': avg_variance,
        'selected_genes': selected_genes,
        'n_bootstraps': n_bootstraps
    }
    
    return results

def plot_bootstrap_results(results, optimal_alpha=0.3):
    """
    Create plots from bootstrap analysis results
    
    Parameters:
    results: Dictionary with computed results from compute_bootstrap_coefficients
    optimal_alpha: Optimal alpha value to mark on plots
    """
    # Unpack results
    coefficient_profiles = results['coefficient_profiles']
    alphas = results['alphas']
    non_zero_counts = results['non_zero_counts']
    avg_variance = results['avg_variance']
    selected_genes = results['selected_genes']
    n_bootstraps = results['n_bootstraps']
    
    # Create the coefficient profile plot
    plt.figure(figsize=(14, 10))
    
    # Plot each selected gene
    n_cols = 4
    n_rows = int(np.ceil(len(selected_genes) / n_cols))
    
    for idx, gene in enumerate(selected_genes):
        plt.subplot(n_rows, n_cols, idx + 1)
        
        # Calculate mean and standard deviation across bootstraps
        mean_coef = np.mean(coefficient_profiles[gene], axis=0)
        std_coef = np.std(coefficient_profiles[gene], axis=0)
        
        # Plot mean coefficient profile
        plt.semilogx(alphas, mean_coef, 'b-', linewidth=2, label='Mean coefficient')
        
        # Add error bars (shaded area for standard deviation)
        plt.fill_between(alphas, mean_coef - std_coef, mean_coef + std_coef, 
                        alpha=0.3, color='blue', label='±1 SD')
        
        # Mark the optimal alpha
        plt.axvline(optimal_alpha, color='red', linestyle='--', linewidth=1.5, 
                   label=f'Optimal alpha ({optimal_alpha})')
        
        plt.xlabel('Alpha (log scale)')
        plt.ylabel('Coefficient value')
        plt.title(f'Gene: {gene}')
        plt.grid(True, which="both", ls="--", alpha=0.3)
        
        # Add legend only to the first subplot
        if idx == 0:
            plt.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.suptitle(f'Lasso Coefficient Profiles with Bootstrap Variability ({n_bootstraps} iterations)\n'
                'Effect of Regularization on Bias and Variance', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.show()
    
    # Additional analysis: Show overall trends
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Average number of non-zero coefficients vs alpha
    plt.subplot(1, 2, 1)
    plt.semilogx(alphas, non_zero_counts, 'g-', linewidth=2)
    plt.axvline(optimal_alpha, color='red', linestyle='--', label='Optimal alpha')
    plt.xlabel('Alpha (log scale)')
    plt.ylabel('Average number of non-zero coefficients')
    plt.title('Model Sparsity vs Regularization')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)
    
    # Plot 2: Average coefficient variance vs alpha
    plt.subplot(1, 2, 2)
    plt.semilogx(alphas, avg_variance, 'r-', linewidth=2)
    plt.axvline(optimal_alpha, color='red', linestyle='--', label='Optimal alpha')
    plt.xlabel('Alpha (log scale)')
    plt.ylabel('Average coefficient variance')
    plt.title('Parameter Variance vs Regularization')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)
    
    plt.tight_layout()
    plt.show()