"""Result analysis tools for cooking conversation experiments."""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np


class ResultAnalyzer:
    """Comprehensive result analysis for cooking conversation experiments."""
    
    def __init__(self, results_file: str = None, results_data: List[Dict] = None):
        """Initialize with either a results file or raw results data."""
        if results_file:
            self.results_data = self._load_results(results_file)
        elif results_data:
            self.results_data = results_data
        else:
            raise ValueError("Must provide either results_file or results_data")
        
        self.df = self._create_dataframe()
    
    def _load_results(self, results_file: str) -> List[Dict]:
        """Load results from JSON file."""
        with open(results_file, 'r') as f:
            data = json.load(f)
            return data.get('results', [])
    
    def _create_dataframe(self) -> pd.DataFrame:
        """Create pandas DataFrame from results for easier analysis."""
        if not self.results_data:
            return pd.DataFrame()
        
        # Flatten nested data
        flattened_results = []
        for result in self.results_data:
            flat_result = {
                'test_id': result.get('test_id'),
                'scenario_name': result.get('scenario_name'),
                'conversation_type': result.get('conversation_type'),
                'llm_provider': result.get('llm_provider'),
                'user_query': result.get('user_query'),
                'success': result.get('success', False),
                'duration_seconds': result.get('duration_seconds'),
                'error_message': result.get('error_message'),
                'total_turns': result.get('metrics', {}).get('total_turns', 0),
                'chef_turns': result.get('metrics', {}).get('chef_turns', 0),
                'trainee_turns': result.get('metrics', {}).get('trainee_turns', 0),
                'outcomes_achieved_count': len(result.get('outcomes_achieved', [])),
                'expected_outcomes_count': len(result.get('expected_outcomes', [])),
                'outcome_completion_rate': len(result.get('outcomes_achieved', [])) / max(1, len(result.get('expected_outcomes', []))),
            }
            
            # Add user profile info
            user_profile = result.get('user_profile', {})
            flat_result['experience_level'] = user_profile.get('experience_level')
            flat_result['has_allergies'] = len(user_profile.get('allergies', [])) > 0
            flat_result['has_dietary_restrictions'] = len(user_profile.get('dietary_restrictions', [])) > 0
            
            # Add metadata
            metadata = result.get('metadata', {})
            flat_result.update({f'meta_{k}': v for k, v in metadata.items()})
            
            flattened_results.append(flat_result)
        
        return pd.DataFrame(flattened_results)
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary report."""
        if self.df.empty:
            return {"error": "No data to analyze"}
        
        total_tests = len(self.df)
        successful_tests = self.df['success'].sum()
        
        summary = {
            "overview": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": total_tests - successful_tests,
                "overall_success_rate": successful_tests / total_tests if total_tests > 0 else 0,
                "average_duration": self.df['duration_seconds'].mean(),
                "average_turns": self.df['total_turns'].mean(),
                "average_outcome_completion": self.df['outcome_completion_rate'].mean()
            },
            "by_conversation_type": self._analyze_by_group('conversation_type'),
            "by_llm_provider": self._analyze_by_group('llm_provider'),
            "by_experience_level": self._analyze_by_group('experience_level'),
            "performance_metrics": self._analyze_performance(),
            "failure_analysis": self._analyze_failures(),
            "conversation_quality": self._analyze_conversation_quality()
        }
        
        return summary
    
    def _analyze_by_group(self, group_column: str) -> Dict[str, Any]:
        """Analyze results grouped by a specific column."""
        if group_column not in self.df.columns:
            return {}
        
        grouped = self.df.groupby(group_column).agg({
            'success': ['count', 'sum', 'mean'],
            'duration_seconds': 'mean',
            'total_turns': 'mean',
            'outcome_completion_rate': 'mean'
        }).round(3)
        
        # Flatten column names
        grouped.columns = ['_'.join(col).strip() for col in grouped.columns]
        
        return grouped.to_dict('index')
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance metrics."""
        performance = {}
        
        if 'duration_seconds' in self.df.columns:
            performance['duration'] = {
                'mean': self.df['duration_seconds'].mean(),
                'median': self.df['duration_seconds'].median(),
                'std': self.df['duration_seconds'].std(),
                'min': self.df['duration_seconds'].min(),
                'max': self.df['duration_seconds'].max()
            }
        
        if 'total_turns' in self.df.columns:
            performance['turns'] = {
                'mean': self.df['total_turns'].mean(),
                'median': self.df['total_turns'].median(),
                'std': self.df['total_turns'].std(),
                'min': self.df['total_turns'].min(),
                'max': self.df['total_turns'].max()
            }
        
        # Efficiency: success per unit time
        successful_df = self.df[self.df['success'] == True]
        if not successful_df.empty and 'duration_seconds' in successful_df.columns:
            performance['efficiency'] = {
                'successful_tests_per_minute': len(successful_df) / (successful_df['duration_seconds'].sum() / 60),
                'average_success_duration': successful_df['duration_seconds'].mean(),
                'fastest_success': successful_df['duration_seconds'].min(),
                'slowest_success': successful_df['duration_seconds'].max()
            }
        
        return performance
    
    def _analyze_failures(self) -> Dict[str, Any]:
        """Analyze failure patterns."""
        failed_df = self.df[self.df['success'] == False]
        
        if failed_df.empty:
            return {"message": "No failures to analyze"}
        
        failure_analysis = {
            "total_failures": len(failed_df),
            "failure_rate": len(failed_df) / len(self.df),
            "failures_by_conversation_type": failed_df['conversation_type'].value_counts().to_dict(),
            "failures_by_llm_provider": failed_df['llm_provider'].value_counts().to_dict(),
            "common_error_patterns": self._categorize_errors(failed_df['error_message'])
        }
        
        return failure_analysis
    
    def _categorize_errors(self, error_messages: pd.Series) -> Dict[str, int]:
        """Categorize error messages into common patterns."""
        error_categories = {
            "api_key_missing": 0,
            "timeout": 0,
            "rate_limit": 0,
            "network_error": 0,
            "authentication": 0,
            "unknown": 0
        }
        
        for error in error_messages.dropna():
            error_lower = str(error).lower()
            
            if "api key" in error_lower or "authentication" in error_lower:
                error_categories["api_key_missing"] += 1
            elif "timeout" in error_lower:
                error_categories["timeout"] += 1
            elif "rate limit" in error_lower or "quota" in error_lower:
                error_categories["rate_limit"] += 1
            elif "network" in error_lower or "connection" in error_lower:
                error_categories["network_error"] += 1
            elif "auth" in error_lower:
                error_categories["authentication"] += 1
            else:
                error_categories["unknown"] += 1
        
        return error_categories
    
    def _analyze_conversation_quality(self) -> Dict[str, Any]:
        """Analyze conversation quality metrics."""
        successful_df = self.df[self.df['success'] == True]
        
        if successful_df.empty:
            return {"message": "No successful conversations to analyze"}
        
        quality_metrics = {
            "average_outcome_completion": successful_df['outcome_completion_rate'].mean(),
            "high_quality_conversations": len(successful_df[successful_df['outcome_completion_rate'] >= 0.8]),
            "conversation_length_distribution": {
                "short_conversations": len(successful_df[successful_df['total_turns'] <= 5]),
                "medium_conversations": len(successful_df[(successful_df['total_turns'] > 5) & (successful_df['total_turns'] <= 10)]),
                "long_conversations": len(successful_df[successful_df['total_turns'] > 10])
            }
        }
        
        return quality_metrics
    
    def create_visualizations(self, output_dir: str = "analysis_plots"):
        """Create visualization plots for the results."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if self.df.empty:
            print("No data to visualize")
            return
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Success rate by conversation type
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        success_by_type = self.df.groupby('conversation_type')['success'].mean()
        success_by_type.plot(kind='bar', title='Success Rate by Conversation Type')
        plt.ylabel('Success Rate')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 2. Success rate by LLM provider
        plt.subplot(1, 2, 2)
        success_by_provider = self.df.groupby('llm_provider')['success'].mean()
        success_by_provider.plot(kind='bar', title='Success Rate by LLM Provider')
        plt.ylabel('Success Rate')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(output_path / 'success_rates.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Duration analysis
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        self.df.boxplot(column='duration_seconds', by='conversation_type', ax=plt.gca())
        plt.title('Duration by Conversation Type')
        plt.ylabel('Duration (seconds)')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        self.df.boxplot(column='duration_seconds', by='llm_provider', ax=plt.gca())
        plt.title('Duration by LLM Provider')
        plt.ylabel('Duration (seconds)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path / 'duration_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Conversation quality heatmap
        if len(self.df) > 10:  # Only create if we have enough data
            plt.figure(figsize=(10, 8))
            
            quality_matrix = self.df.pivot_table(
                values='outcome_completion_rate',
                index='conversation_type',
                columns='llm_provider',
                aggfunc='mean'
            )
            
            sns.heatmap(quality_matrix, annot=True, cmap='RdYlGn', 
                       center=0.5, vmin=0, vmax=1)
            plt.title('Outcome Completion Rate by Type and Provider')
            plt.tight_layout()
            plt.savefig(output_path / 'quality_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 5. Turn count distribution
        plt.figure(figsize=(10, 6))
        plt.hist(self.df['total_turns'], bins=20, alpha=0.7, edgecolor='black')
        plt.title('Distribution of Conversation Turn Counts')
        plt.xlabel('Total Turns')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(output_path / 'turn_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualizations saved to: {output_path}")
    
    def export_detailed_csv(self, output_file: str = "detailed_results.csv"):
        """Export detailed results to CSV for further analysis."""
        self.df.to_csv(output_file, index=False)
        print(f"📄 Detailed results exported to: {output_file}")
    
    def compare_providers(self, providers: List[str] = None) -> pd.DataFrame:
        """Compare performance across LLM providers."""
        if providers:
            df_filtered = self.df[self.df['llm_provider'].isin(providers)]
        else:
            df_filtered = self.df
        
        comparison = df_filtered.groupby('llm_provider').agg({
            'success': ['count', 'sum', 'mean'],
            'duration_seconds': ['mean', 'median', 'std'],
            'total_turns': ['mean', 'std'],
            'outcome_completion_rate': ['mean', 'std']
        }).round(3)
        
        return comparison
    
    def find_best_performers(self, metric: str = 'success_rate', top_n: int = 5) -> pd.DataFrame:
        """Find best performing scenario-provider combinations."""
        if metric == 'success_rate':
            grouped = self.df.groupby(['scenario_name', 'llm_provider'])['success'].mean()
        elif metric == 'outcome_completion':
            grouped = self.df.groupby(['scenario_name', 'llm_provider'])['outcome_completion_rate'].mean()
        elif metric == 'efficiency':
            # Success rate divided by average duration
            grouped = self.df.groupby(['scenario_name', 'llm_provider']).apply(
                lambda x: x['success'].mean() / (x['duration_seconds'].mean() / 60)
            )
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return grouped.sort_values(ascending=False).head(top_n).reset_index()


def analyze_results_cli():
    """CLI function for analyzing results."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze cooking conversation test results")
    parser.add_argument("results_file", help="Path to results JSON file")
    parser.add_argument("--output-dir", default="analysis_output", help="Output directory for analysis")
    parser.add_argument("--create-plots", action="store_true", help="Create visualization plots")
    parser.add_argument("--export-csv", action="store_true", help="Export detailed CSV")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = ResultAnalyzer(results_file=args.results_file)
    
    # Generate summary report
    summary = analyzer.generate_summary_report()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save summary report
    with open(output_path / "summary_report.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Print key findings
    print("📊 ANALYSIS SUMMARY")
    print("=" * 50)
    
    overview = summary.get('overview', {})
    print(f"Total tests: {overview.get('total_tests', 0)}")
    print(f"Success rate: {overview.get('overall_success_rate', 0):.1%}")
    print(f"Average duration: {overview.get('average_duration', 0):.1f}s")
    print(f"Average turns: {overview.get('average_turns', 0):.1f}")
    
    # Provider comparison
    print(f"\n🤖 LLM PROVIDER PERFORMANCE:")
    provider_stats = summary.get('by_llm_provider', {})
    for provider, stats in provider_stats.items():
        success_rate = stats.get('success_mean', 0)
        test_count = stats.get('success_count', 0)
        print(f"  {provider}: {success_rate:.1%} ({test_count} tests)")
    
    # Create visualizations if requested
    if args.create_plots:
        analyzer.create_visualizations(output_path / "plots")
    
    # Export CSV if requested
    if args.export_csv:
        analyzer.export_detailed_csv(output_path / "detailed_results.csv")
    
    print(f"\n📁 Analysis saved to: {output_path}")


if __name__ == "__main__":
    analyze_results_cli()