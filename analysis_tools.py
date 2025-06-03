import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import numpy as np
from pathlib import Path

class AblationAnalyzer:
    """Analyze ablation study results"""
    
    def __init__(self, results_df: pd.DataFrame):
        self.df = results_df
        
    def compare_configurations(self, metrics: List[str] = None) -> pd.DataFrame:
        """Compare different configurations across key metrics"""
        if metrics is None:
            metrics = ["task_success", "total_tokens", "total_cost", "conversation_length", "clarification_requests"]
        
        comparison = self.df.groupby("config_name")[metrics].agg(['mean', 'std']).round(4)
        return comparison
    
    def feature_impact_analysis(self) -> Dict[str, float]:
        """Analyze impact of individual features"""
        baseline = self.df[self.df["config_name"] == "baseline"]
        if baseline.empty:
            print("Warning: No baseline configuration found")
            return {}
        
        baseline_success = baseline["task_success"].mean()
        
        impacts = {}
        feature_columns = ["use_faiss_search", "use_ontology_substitution", "use_llm_validation"]
        
        for feature in feature_columns:
            # Compare configs with/without this feature
            with_feature = self.df[self.df[feature] == True]["task_success"].mean()
            without_feature = self.df[self.df[feature] == False]["task_success"].mean()
            
            impacts[feature] = with_feature - without_feature
        
        return impacts
    
    def cost_efficiency_analysis(self) -> pd.DataFrame:
        """Analyze cost vs performance trade-offs"""
        efficiency = self.df.groupby("config_name").agg({
            "task_success": "mean",
            "total_cost": "mean",
            "total_tokens": "mean"
        }).reset_index()
        
        # Calculate efficiency ratio (success per dollar)
        efficiency["success_per_dollar"] = efficiency["task_success"] / (efficiency["total_cost"] + 0.001)  # Avoid division by zero
        efficiency["success_per_token"] = efficiency["task_success"] / (efficiency["total_tokens"] + 1)
        
        return efficiency.sort_values("success_per_dollar", ascending=False)
    
    def plot_configuration_comparison(self, metric: str = "task_success", save_path: str = None):
        """Plot comparison of configurations"""
        plt.figure(figsize=(12, 6))
        
        # Box plot showing distribution
        sns.boxplot(data=self.df, x="config_name", y=metric)
        plt.xticks(rotation=45)
        plt.title(f"Configuration Comparison: {metric}")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_feature_impact(self, save_path: str = None):
        """Plot impact of individual features"""
        impacts = self.feature_impact_analysis()
        
        plt.figure(figsize=(10, 6))
        features = list(impacts.keys())
        values = list(impacts.values())
        
        colors = ['green' if v > 0 else 'red' for v in values]
        bars = plt.bar(features, values, color=colors, alpha=0.7)
        
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title("Feature Impact on Task Success")
        plt.ylabel("Impact on Success Rate")
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_cost_vs_performance(self, save_path: str = None):
        """Plot cost vs performance scatter"""
        efficiency = self.cost_efficiency_analysis()
        
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(efficiency["total_cost"], efficiency["task_success"], 
                            s=100, alpha=0.7, c=range(len(efficiency)), cmap='viridis')
        
        # Add labels for each point
        for i, row in efficiency.iterrows():
            plt.annotate(row["config_name"], 
                        (row["total_cost"], row["task_success"]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        plt.xlabel("Average Total Cost ($)")
        plt.ylabel("Task Success Rate")
        plt.title("Cost vs Performance Trade-off")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def generate_report(self, output_dir: str = "analysis_output") -> str:
        """Generate comprehensive analysis report"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        report = []
        report.append("# Ablation Study Analysis Report\n")
        report.append(f"Generated: {pd.Timestamp.now()}\n")
        report.append(f"Total experiments: {len(self.df)}\n")
        report.append(f"Configurations tested: {self.df['config_name'].nunique()}\n\n")
        
        # Configuration comparison
        report.append("## Configuration Comparison\n")
        comparison = self.compare_configurations()
        report.append(comparison.to_string())
        report.append("\n\n")
        
        # Feature impact
        report.append("## Feature Impact Analysis\n")
        impacts = self.feature_impact_analysis()
        for feature, impact in impacts.items():
            report.append(f"- {feature}: {impact:+.3f}\n")
        report.append("\n")
        
        # Cost efficiency
        report.append("## Cost Efficiency Ranking\n")
        efficiency = self.cost_efficiency_analysis()
        for i, row in efficiency.head().iterrows():
            report.append(f"{i+1}. {row['config_name']}: {row['success_per_dollar']:.3f} success/$\n")
        report.append("\n")
        
        # Recommendations
        report.append("## Recommendations\n")
        best_config = efficiency.iloc[0]["config_name"]
        report.append(f"- **Most cost-efficient**: {best_config}\n")
        
        best_performance = self.df.groupby("config_name")["task_success"].mean().idxmax()
        report.append(f"- **Best performance**: {best_performance}\n")
        
        # Find most impactful feature
        most_impactful = max(impacts.items(), key=lambda x: abs(x[1]))
        report.append(f"- **Most impactful feature**: {most_impactful[0]} ({most_impactful[1]:+.3f})\n")
        
        # Save report
        report_text = "".join(report)
        report_file = output_path / "ablation_report.md"
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        # Generate plots
        self.plot_configuration_comparison(save_path=output_path / "config_comparison.png")
        self.plot_feature_impact(save_path=output_path / "feature_impact.png")
        self.plot_cost_vs_performance(save_path=output_path / "cost_performance.png")
        
        print(f"Analysis report saved to {report_file}")
        return str(report_file)

# Statistical significance testing
def statistical_significance_test(df: pd.DataFrame, config1: str, config2: str, metric: str = "task_success"):
    """Test statistical significance between two configurations"""
    from scipy import stats
    
    group1 = df[df["config_name"] == config1][metric]
    group2 = df[df["config_name"] == config2][metric]
    
    # Perform t-test
    statistic, p_value = stats.ttest_ind(group1, group2)
    
    return {
        "config1": config1,
        "config2": config2,
        "metric": metric,
        "config1_mean": group1.mean(),
        "config2_mean": group2.mean(),
        "difference": group1.mean() - group2.mean(),
        "t_statistic": statistic,
        "p_value": p_value,
        "significant": p_value < 0.05
    }

if __name__ == "__main__":
    # Example usage
    try:
        df = pd.read_csv("experiment_results/ablation_study_results.csv")
        analyzer = AblationAnalyzer(df)
        
        # Generate comprehensive analysis
        report_path = analyzer.generate_report()
        print(f"Analysis complete. Report saved at: {report_path}")
        
        # Example significance test
        if len(df["config_name"].unique()) >= 2:
            configs = df["config_name"].unique()[:2]
            sig_test = statistical_significance_test(df, configs[0], configs[1])
            print(f"Significance test: {sig_test}")
    
    except FileNotFoundError:
        print("No results file found. Run experiment_runner.py first.")