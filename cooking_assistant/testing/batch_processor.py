"""Batch processing utilities for running multiple conversation experiments."""

import json
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import multiprocessing
import queue
import threading


@dataclass
class BatchConfig:
    """Configuration for batch test execution."""
    max_workers: int = 3
    timeout_per_test: int = 300
    retry_failed: bool = True
    max_retries: int = 2
    save_intermediate: bool = True
    intermediate_save_interval: int = 10
    use_multiprocessing: bool = False  # Thread vs Process based parallelism


class BatchProcessor:
    """Enhanced batch processor with progress tracking and recovery."""
    
    def __init__(self, config: BatchConfig):
        self.config = config
        self.results_queue = queue.Queue()
        self.progress_callback: Optional[Callable] = None
        self.stop_event = threading.Event()
        
    def set_progress_callback(self, callback: Callable[[int, int, str], None]):
        """Set callback for progress updates: callback(completed, total, current_task)."""
        self.progress_callback = callback
    
    def run_batch_with_recovery(self, test_runner, scenarios, llm_providers, output_file: str) -> List:
        """Run batch tests with progress tracking and intermediate saves."""
        
        # Create all test combinations
        test_tasks = []
        for scenario in scenarios:
            for provider in llm_providers:
                if provider in scenario.llm_providers:
                    test_tasks.append((scenario, provider))
        
        total_tests = len(test_tasks)
        completed_tests = 0
        results = []
        failed_tasks = []
        
        print(f"🚀 Starting batch processing: {total_tests} tests with {self.config.max_workers} workers")
        
        # Load existing results if continuing from a previous run
        existing_results = self._load_existing_results(output_file)
        if existing_results:
            print(f"📂 Found {len(existing_results)} existing results, resuming...")
            results.extend(existing_results)
            completed_test_ids = {r.get('test_id') for r in existing_results if 'test_id' in r}
            test_tasks = [(s, p) for s, p in test_tasks 
                         if f"{s.name}_{p.value}" not in completed_test_ids]
        
        start_time = time.time()
        
        executor_class = ProcessPoolExecutor if self.config.use_multiprocessing else ThreadPoolExecutor
        
        with executor_class(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_task = {}
            for scenario, provider in test_tasks:
                future = executor.submit(
                    self._run_single_test_with_retry,
                    test_runner, scenario, provider
                )
                future_to_task[future] = (scenario, provider)
            
            # Process completed tests
            for future in as_completed(future_to_task):
                if self.stop_event.is_set():
                    print("🛑 Stopping batch processing...")
                    break
                
                scenario, provider = future_to_task[future]
                
                try:
                    result = future.result(timeout=self.config.timeout_per_test)
                    results.append(result.to_dict() if hasattr(result, 'to_dict') else result)
                    
                    completed_tests += 1
                    status = "✅ PASS" if result.success else "❌ FAIL"
                    elapsed = time.time() - start_time
                    
                    print(f"{status} [{completed_tests}/{total_tests}] {scenario.name} ({provider.value}) - {elapsed:.1f}s")
                    
                    # Progress callback
                    if self.progress_callback:
                        self.progress_callback(completed_tests, total_tests, f"{scenario.name}_{provider.value}")
                    
                    # Intermediate save
                    if (self.config.save_intermediate and 
                        completed_tests % self.config.intermediate_save_interval == 0):
                        self._save_intermediate_results(results, output_file, completed_tests, total_tests)
                
                except Exception as e:
                    failed_tasks.append((scenario, provider, str(e)))
                    print(f"❌ ERROR {scenario.name} ({provider.value}): {e}")
                    completed_tests += 1
        
        # Handle failed tasks with retries
        if failed_tasks and self.config.retry_failed:
            print(f"🔄 Retrying {len(failed_tasks)} failed tests...")
            retry_results = self._retry_failed_tests(test_runner, failed_tasks)
            results.extend(retry_results)
        
        # Final save
        self._save_final_results(results, output_file)
        
        elapsed_total = time.time() - start_time
        print(f"🎉 Batch completed in {elapsed_total:.1f}s")
        
        return results
    
    def _run_single_test_with_retry(self, test_runner, scenario, provider):
        """Run a single test with built-in retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                return test_runner.run_single_test(scenario, provider, self.config.timeout_per_test)
            except Exception as e:
                last_exception = e
                if attempt < self.config.max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"⚠️  Retry {attempt + 1}/{self.config.max_retries} for {scenario.name} in {wait_time}s...")
                    time.sleep(wait_time)
        
        # If all retries failed, create a failure result
        from .test_runner import TestResult
        failed_result = TestResult(scenario, provider)
        failed_result.success = False
        failed_result.error_message = f"Failed after {self.config.max_retries} retries: {last_exception}"
        failed_result.start_time = datetime.now()
        failed_result.end_time = datetime.now()
        return failed_result
    
    def _retry_failed_tests(self, test_runner, failed_tasks):
        """Retry failed tests with reduced parallelism."""
        retry_results = []
        retry_workers = max(1, self.config.max_workers // 2)
        
        with ThreadPoolExecutor(max_workers=retry_workers) as executor:
            futures = [
                executor.submit(test_runner.run_single_test, scenario, provider, self.config.timeout_per_test * 2)
                for scenario, provider, _ in failed_tasks
            ]
            
            for future, (scenario, provider, _) in zip(futures, failed_tasks):
                try:
                    result = future.result()
                    retry_results.append(result.to_dict())
                    status = "✅ RETRY SUCCESS" if result.success else "❌ RETRY FAILED"
                    print(f"{status} {scenario.name} ({provider.value})")
                except Exception as e:
                    print(f"❌ RETRY ERROR {scenario.name} ({provider.value}): {e}")
        
        return retry_results
    
    def _load_existing_results(self, output_file: str) -> List[Dict]:
        """Load existing results from file."""
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
                return data.get('results', [])
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def _save_intermediate_results(self, results: List, output_file: str, completed: int, total: int):
        """Save intermediate results."""
        backup_file = output_file.replace('.json', f'_backup_{completed}of{total}.json')
        self._save_results_to_file(results, backup_file, f"intermediate_{completed}of{total}")
        print(f"💾 Intermediate save: {backup_file}")
    
    def _save_final_results(self, results: List, output_file: str):
        """Save final results."""
        self._save_results_to_file(results, output_file, "final")
        print(f"💾 Final results saved: {output_file}")
    
    def _save_results_to_file(self, results: List, filename: str, batch_type: str):
        """Save results to file with metadata."""
        results_data = {
            "batch_type": batch_type,
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(results),
            "successful_tests": len([r for r in results if r.get('success', False)]),
            "failed_tests": len([r for r in results if not r.get('success', True)]),
            "results": results
        }
        
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)


class ProgressTracker:
    """Progress tracking utility for long-running batch jobs."""
    
    def __init__(self):
        self.start_time = time.time()
        self.last_update_time = time.time()
        self.completed_tests = 0
        self.total_tests = 0
        self.current_task = ""
    
    def update(self, completed: int, total: int, current_task: str = ""):
        """Update progress."""
        self.completed_tests = completed
        self.total_tests = total
        self.current_task = current_task
        self.last_update_time = time.time()
        
        # Calculate metrics
        elapsed = self.last_update_time - self.start_time
        if completed > 0:
            avg_time_per_test = elapsed / completed
            remaining_tests = total - completed
            eta = remaining_tests * avg_time_per_test
        else:
            eta = 0
        
        # Print progress
        percentage = (completed / total * 100) if total > 0 else 0
        print(f"📊 Progress: {completed}/{total} ({percentage:.1f}%) | "
              f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s | Current: {current_task}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get progress summary."""
        total_elapsed = time.time() - self.start_time
        return {
            "completed_tests": self.completed_tests,
            "total_tests": self.total_tests,
            "percentage_complete": (self.completed_tests / self.total_tests * 100) if self.total_tests > 0 else 0,
            "total_elapsed_seconds": total_elapsed,
            "average_time_per_test": total_elapsed / self.completed_tests if self.completed_tests > 0 else 0,
            "current_task": self.current_task
        }


def run_batch_experiments(scenarios, llm_providers, output_dir: str = "test_results", batch_name: str = None, **kwargs):
    """Simplified function to run batch experiments with sensible defaults."""
    
    if not batch_name:
        batch_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create batch config
    batch_config = BatchConfig(
        max_workers=kwargs.get('max_workers', 3),
        timeout_per_test=kwargs.get('timeout', 300),
        retry_failed=kwargs.get('retry_failed', True),
        save_intermediate=kwargs.get('save_intermediate', True)
    )
    
    # Initialize components
    from .test_runner import ConversationTestRunner
    
    test_runner = ConversationTestRunner(
        recipes_csv_path=kwargs.get('recipes_csv', '13k-recipes.csv'),
        sample_size=kwargs.get('sample_size', 100)
    )
    
    processor = BatchProcessor(batch_config)
    tracker = ProgressTracker()
    processor.set_progress_callback(tracker.update)
    
    # Run batch  
    # Generate descriptive filename preview for batch
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    providers_preview = '+'.join([p.value.replace('_', '') for p in llm_providers[:3]])
    if len(llm_providers) > 3:
        providers_preview += f"+{len(llm_providers)-3}more"
    
    types_preview = '+'.join([t.value.replace('_', '-') for t in set([s.conversation_type for s in scenarios])[:3]])
    preview_filename = f"{timestamp}_{providers_preview}_{types_preview}_{batch_name}_inprogress.json"
    
    output_file = Path(output_dir) / preview_filename
    results = processor.run_batch_with_recovery(
        test_runner, scenarios, llm_providers, str(output_file)
    )
    
    # Generate final results with proper naming
    from .test_runner import TestResult
    if results and hasattr(results[0], 'to_dict'):
        # Convert to TestResult objects if needed
        test_results = results if isinstance(results[0], TestResult) else [TestResult.from_dict(r) for r in results]
        final_output_file = test_runner.save_results(test_results, batch_name)
    else:
        final_output_file = str(output_file)
    
    # Generate summary
    summary = tracker.get_summary()
    print(f"\n📋 BATCH SUMMARY:")
    print(f"  Completed: {summary['completed_tests']}/{summary['total_tests']}")
    print(f"  Success rate: {len([r for r in results if r.get('success')])/len(results):.1%}")
    print(f"  Total time: {summary['total_elapsed_seconds']:.1f}s")
    print(f"  Avg per test: {summary['average_time_per_test']:.1f}s")
    print(f"  Final results: {final_output_file}")
    
    return results, final_output_file