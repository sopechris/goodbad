import pandas as pd
import re
import hashlib
import time

class LocalDuplicateIdentifier:
    def __init__(self):
        self.similarity_threshold = 0.85

    def normalize_text(self, text):
        """Ultra-fast text normalization."""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        text = text.lower().strip()
        # Only essential normalization for speed
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[.!?;,]+$', '', text)
        
        return text
    
    def get_text_hash(self, text):
        """Generate hash for exact duplicate detection."""
        normalized = self.normalize_text(text)
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def find_exact_duplicates(self, df, school_column='school', text_column='text'):
        """Only exact duplicate detection."""
        start_time = time.time()
        print(f"Starting exact duplicate detection only")
        print(f"Dataset: {len(df)} rows")

        all_duplicates = []
        for school in df[school_column].dropna().unique():
            school_df = df[df[school_column] == school].copy()
            if len(school_df) < 2:
                continue
            # Fast hash-based duplicate detection
            school_df['text_hash'] = school_df[text_column].apply(self.get_text_hash)
            hash_counts = school_df['text_hash'].value_counts()
            duplicate_hashes = hash_counts[hash_counts > 1].index
            
            for text_hash in duplicate_hashes:
                duplicate_rows = school_df[school_df['text_hash'] == text_hash]
                original_idx = duplicate_rows.index[0]
                original_text = df.loc[original_idx, text_column]
                
                for dup_idx in duplicate_rows.index[1:]:
                    dup_text = df.loc[dup_idx, text_column]
                    all_duplicates.append({
                        'index': dup_idx,
                        'text': dup_text,
                        'school': school,
                        'duplicate_type': 'exact',
                        'duplicate_of_index': original_idx,
                        'duplicate_of_text': original_text,
                        'similarity_score': 1.0
                    })

        end_time = time.time()
        print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")
        print(f"Total exact duplicates found: {len(all_duplicates)}")
        return all_duplicates

def main():
    finder = LocalDuplicateIdentifier()
    try:
        df = pd.read_csv("religion_data.csv")
        print(f"Loaded {len(df)} rows")
        if 'text' not in df.columns or 'school' not in df.columns:
            print("Error: Required columns missing!")
            return
    except Exception as e:
        print(f"Error: {e}")
        return

    duplicates = finder.find_exact_duplicates(df, 'school', 'text')
    if duplicates:
        duplicate_df = pd.DataFrame(duplicates)
        duplicate_df.to_csv("duplicates_dropped.csv", index=False)
        print(f"\nSaved {len(duplicates)} duplicates")
        duplicate_indices = set(dup['index'] for dup in duplicates)
        clean_df = df[~df.index.isin(duplicate_indices)].copy()
        clean_df.to_csv("religion_data_cleaned.csv", index=False)
        print(f"Saved {len(clean_df)} clean rows")

if __name__ == "__main__":
    main()