#!/usr/bin/env python3
"""
Migration script to rename existing JSON files with the new informative naming scheme.

This script will:
1. Scan for existing JSON files with old naming patterns
2. Extract metadata from file contents
3. Rename files using the new informative scheme
4. Create a backup mapping file
"""

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import uuid

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class JSONFileMigrator:
    """Migrates JSON files to new naming scheme."""
    
    def __init__(self, source_dir: str = "."):
        self.source_dir = Path(source_dir)
        self.backup_mappings = []
        
    def migrate_all_files(self, dry_run: bool = True):
        """Migrate all JSON files in the directory."""
        print(f"🔍 Scanning {self.source_dir} for JSON files...")
        
        # Find different types of JSON files
        cooking_session_files = list(self.source_dir.glob("cooking_session_*.json"))
        multi_agent_files = list(self.source_dir.glob("multi_agent_cooking_conversation_*.json"))
        token_cost_files = list(self.source_dir.glob("token_cost_log_*.json"))
        test_result_files = list(self.source_dir.glob("test_*.json"))
        
        print(f"Found {len(cooking_session_files)} cooking session files")
        print(f"Found {len(multi_agent_files)} multi-agent conversation files")
        print(f"Found {len(token_cost_files)} token cost log files")
        print(f"Found {len(test_result_files)} test result files")
        
        total_files = len(cooking_session_files) + len(multi_agent_files) + len(token_cost_files) + len(test_result_files)
        
        if total_files == 0:
            print("No files to migrate.")
            return
        
        if dry_run:
            print(f"\n🧪 DRY RUN MODE - No files will be moved")
        else:
            print(f"\n📝 MIGRATION MODE - Files will be renamed")
        
        # Migrate each type
        self._migrate_cooking_sessions(cooking_session_files, dry_run)
        self._migrate_multi_agent_conversations(multi_agent_files, dry_run)
        self._migrate_token_cost_logs(token_cost_files, dry_run)
        self._migrate_test_results(test_result_files, dry_run)
        
        # Save backup mapping
        if not dry_run and self.backup_mappings:
            self._save_backup_mapping()
        
        print(f"\n✅ Migration complete! {len(self.backup_mappings)} files processed.")
    
    def _migrate_cooking_sessions(self, files: List[Path], dry_run: bool):
        """Migrate cooking session files."""
        print(f"\n📋 Migrating cooking session files...")
        
        for file_path in files:
            try:
                metadata = self._extract_cooking_session_metadata(file_path)
                new_name = self._generate_cooking_session_name(metadata, file_path)
                self._rename_file(file_path, new_name, dry_run)
                
            except Exception as e:
                print(f"❌ Error processing {file_path.name}: {e}")
    
    def _migrate_multi_agent_conversations(self, files: List[Path], dry_run: bool):
        """Migrate multi-agent conversation files."""
        print(f"\n💬 Migrating multi-agent conversation files...")
        
        for file_path in files:
            try:
                metadata = self._extract_multi_agent_metadata(file_path)
                new_name = self._generate_multi_agent_name(metadata, file_path)
                self._rename_file(file_path, new_name, dry_run)
                
            except Exception as e:
                print(f"❌ Error processing {file_path.name}: {e}")
    
    def _migrate_token_cost_logs(self, files: List[Path], dry_run: bool):
        """Migrate token cost log files."""
        print(f"\n💰 Migrating token cost log files...")
        
        for file_path in files:
            try:
                metadata = self._extract_token_cost_metadata(file_path)
                new_name = self._generate_token_cost_name(metadata, file_path)
                self._rename_file(file_path, new_name, dry_run)
                
            except Exception as e:
                print(f"❌ Error processing {file_path.name}: {e}")
    
    def _migrate_test_results(self, files: List[Path], dry_run: bool):
        """Migrate test result files."""
        print(f"\n🧪 Migrating test result files...")
        
        for file_path in files:
            try:
                # These might already be in good format, check first
                if self._is_already_new_format(file_path.name):
                    print(f"⏭️  Skipping {file_path.name} (already new format)")
                    continue
                    
                metadata = self._extract_test_result_metadata(file_path)
                new_name = self._generate_test_result_name(metadata, file_path)
                self._rename_file(file_path, new_name, dry_run)
                
            except Exception as e:
                print(f"❌ Error processing {file_path.name}: {e}")
    
    def _extract_cooking_session_metadata(self, file_path: Path) -> Dict:
        """Extract metadata from cooking session file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract job ID from filename
        job_id_match = re.search(r'cooking_session_([a-f0-9-]+)', file_path.name)
        job_id = job_id_match.group(1) if job_id_match else str(uuid.uuid4())[:8]
        
        # Try to infer LLM provider from conversation content
        conversation = data.get("conversation", [])
        llm_provider = self._infer_llm_provider(conversation)
        
        # Try to infer conversation type from content
        conv_type = self._infer_conversation_type(conversation)
        
        # Get file modification time as timestamp
        timestamp = datetime.fromtimestamp(file_path.stat().st_mtime).strftime('%Y%m%d_%H%M%S')
        
        return {
            "job_id": job_id[:8],
            "llm_provider": llm_provider,
            "conversation_type": conv_type,
            "timestamp": timestamp,
            "file_type": "cooking-session",
            "original_name": file_path.name
        }
    
    def _extract_multi_agent_metadata(self, file_path: Path) -> Dict:
        """Extract metadata from multi-agent conversation file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract job ID from filename
        job_id_match = re.search(r'multi_agent_cooking_conversation_([a-f0-9-]+)', file_path.name)
        job_id = job_id_match.group(1) if job_id_match else str(uuid.uuid4())[:8]
        
        # Look for LLM provider in the data
        llm_provider = "unknown"
        if "llm_provider" in data:
            llm_provider = data["llm_provider"]
        elif "conversation" in data:
            llm_provider = self._infer_llm_provider(data["conversation"])
        
        timestamp = datetime.fromtimestamp(file_path.stat().st_mtime).strftime('%Y%m%d_%H%M%S')
        
        return {
            "job_id": job_id[:8],
            "llm_provider": llm_provider,
            "conversation_type": "multi-agent",
            "timestamp": timestamp,
            "file_type": "multi-agent-conversation",
            "original_name": file_path.name
        }
    
    def _extract_token_cost_metadata(self, file_path: Path) -> Dict:
        """Extract metadata from token cost log file."""
        # Extract job ID from filename
        job_id_match = re.search(r'token_cost_log_([a-f0-9-]+)', file_path.name)
        job_id = job_id_match.group(1) if job_id_match else str(uuid.uuid4())[:8]
        
        # Check if it's improved version
        is_improved = "_improved" in file_path.name
        
        timestamp = datetime.fromtimestamp(file_path.stat().st_mtime).strftime('%Y%m%d_%H%M%S')
        
        return {
            "job_id": job_id[:8],
            "llm_provider": "unknown",
            "conversation_type": "token-cost",
            "timestamp": timestamp,
            "file_type": "token-cost-log",
            "is_improved": is_improved,
            "original_name": file_path.name
        }
    
    def _extract_test_result_metadata(self, file_path: Path) -> Dict:
        """Extract metadata from test result file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract providers and conversation types from results
            results = data.get("results", [])
            if results:
                providers = list(set([r.get("llm_provider", "unknown") for r in results]))
                conv_types = list(set([r.get("conversation_type", "unknown") for r in results]))
            else:
                providers = ["unknown"]
                conv_types = ["unknown"]
            
            success_rate = 0
            if "successful_tests" in data and "total_tests" in data:
                total = data["total_tests"]
                success = data["successful_tests"]
                success_rate = int((success / total) * 100) if total > 0 else 0
            
            timestamp = data.get("timestamp", datetime.fromtimestamp(file_path.stat().st_mtime).isoformat())
            if isinstance(timestamp, str) and 'T' in timestamp:
                timestamp = datetime.fromisoformat(timestamp).strftime('%Y%m%d_%H%M%S')
            
            return {
                "job_id": str(uuid.uuid4())[:8],
                "llm_providers": providers,
                "conversation_types": conv_types,
                "timestamp": timestamp,
                "file_type": "test-results",
                "success_rate": success_rate,
                "batch_name": data.get("batch_name", "unknown"),
                "original_name": file_path.name
            }
            
        except Exception as e:
            # Fallback for unparseable files
            return {
                "job_id": str(uuid.uuid4())[:8],
                "llm_providers": ["unknown"],
                "conversation_types": ["unknown"],
                "timestamp": datetime.fromtimestamp(file_path.stat().st_mtime).strftime('%Y%m%d_%H%M%S'),
                "file_type": "test-results",
                "success_rate": 0,
                "batch_name": "unknown",
                "original_name": file_path.name
            }
    
    def _infer_llm_provider(self, conversation: List) -> str:
        """Infer LLM provider from conversation content."""
        # Look for provider hints in the conversation
        full_text = " ".join([str(msg.get("content", "")) for msg in conversation]).lower()
        
        if "gpt" in full_text or "openai" in full_text:
            return "openai"
        elif "claude" in full_text or "anthropic" in full_text:
            return "anthropic"
        elif "gemini" in full_text or "google" in full_text:
            return "google"
        else:
            return "unknown"
    
    def _infer_conversation_type(self, conversation: List) -> str:
        """Infer conversation type from content."""
        if not conversation:
            return "unknown"
        
        full_text = " ".join([str(msg.get("content", "")) for msg in conversation]).lower()
        
        if "substitute" in full_text or "replace" in full_text:
            return "substitution"
        elif "allergen" in full_text or "allergy" in full_text:
            return "allergen-safe"
        elif "vegan" in full_text or "keto" in full_text or "gluten" in full_text:
            return "dietary-restriction"
        elif "recipe" in full_text:
            return "specific-recipe"
        else:
            return "general"
    
    def _generate_cooking_session_name(self, metadata: Dict, original_path: Path) -> str:
        """Generate new name for cooking session file."""
        provider = metadata["llm_provider"].replace('_', '')
        conv_type = metadata["conversation_type"].replace('_', '-')
        
        return f"{metadata['timestamp']}_{provider}_{conv_type}_cooking-session_{metadata['job_id']}.json"
    
    def _generate_multi_agent_name(self, metadata: Dict, original_path: Path) -> str:
        """Generate new name for multi-agent conversation file."""
        provider = metadata["llm_provider"].replace('_', '')
        
        return f"{metadata['timestamp']}_{provider}_multi-agent_conversation_{metadata['job_id']}.json"
    
    def _generate_token_cost_name(self, metadata: Dict, original_path: Path) -> str:
        """Generate new name for token cost log file."""
        suffix = "_improved" if metadata.get("is_improved") else ""
        
        return f"{metadata['timestamp']}_token-cost-log_{metadata['job_id']}{suffix}.json"
    
    def _generate_test_result_name(self, metadata: Dict, original_path: Path) -> str:
        """Generate new name for test result file."""
        providers = metadata["llm_providers"]
        conv_types = metadata["conversation_types"]
        
        # Create provider string
        if len(providers) == 1:
            provider_str = providers[0].replace('_', '')
        elif len(providers) <= 3:
            provider_str = '+'.join([p.replace('_', '') for p in providers])
        else:
            provider_str = f"{len(providers)}providers"
        
        # Create conversation type string
        if len(conv_types) == 1:
            type_str = conv_types[0].replace('_', '-')
        elif len(conv_types) <= 3:
            type_str = '+'.join([ct.replace('_', '-') for ct in conv_types])
        else:
            type_str = "all-types"
        
        batch_name = metadata["batch_name"].replace(' ', '-').replace('_', '-')
        success_rate = metadata["success_rate"]
        
        return f"{metadata['timestamp']}_{provider_str}_{type_str}_{batch_name}_{success_rate}pct_{metadata['job_id']}.json"
    
    def _is_already_new_format(self, filename: str) -> bool:
        """Check if filename is already in new format."""
        # New format: timestamp_provider_type_name_metrics_jobid.json
        pattern = r'^\d{8}_\d{4}_[a-zA-Z0-9+]+_[a-zA-Z0-9-+]+_.*_[a-f0-9]{8}\.json$'
        return bool(re.match(pattern, filename))
    
    def _rename_file(self, old_path: Path, new_name: str, dry_run: bool):
        """Rename a file and track the mapping."""
        new_path = old_path.parent / new_name
        
        mapping = {
            "old_name": old_path.name,
            "new_name": new_name,
            "old_path": str(old_path),
            "new_path": str(new_path),
            "timestamp": datetime.now().isoformat()
        }
        
        if dry_run:
            print(f"  📝 {old_path.name} -> {new_name}")
        else:
            if new_path.exists():
                print(f"  ⚠️  Target exists, skipping: {new_name}")
                return
            
            old_path.rename(new_path)
            print(f"  ✅ {old_path.name} -> {new_name}")
        
        self.backup_mappings.append(mapping)
    
    def _save_backup_mapping(self):
        """Save the backup mapping to a file."""
        mapping_file = self.source_dir / f"migration_mapping_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        mapping_data = {
            "migration_date": datetime.now().isoformat(),
            "total_files": len(self.backup_mappings),
            "mappings": self.backup_mappings
        }
        
        with open(mapping_file, 'w') as f:
            json.dump(mapping_data, f, indent=2)
        
        print(f"📁 Migration mapping saved to: {mapping_file}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate JSON files to new naming scheme")
    parser.add_argument("--directory", "-d", default=".", help="Directory to scan for JSON files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be renamed without doing it")
    parser.add_argument("--file-types", nargs="+", 
                      choices=["cooking-session", "multi-agent", "token-cost", "test-results", "all"],
                      default=["all"], help="Types of files to migrate")
    
    args = parser.parse_args()
    
    print("🔄 JSON File Naming Migration Tool")
    print("=" * 50)
    
    migrator = JSONFileMigrator(args.directory)
    migrator.migrate_all_files(dry_run=args.dry_run)
    
    if args.dry_run:
        print(f"\n💡 To actually perform the migration, run again without --dry-run")


if __name__ == "__main__":
    main()