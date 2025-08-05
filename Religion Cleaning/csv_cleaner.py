import pandas as pd
import re
import spacy
from typing import List, Set, Dict
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class AdvancedReligiousTextCleaner:
    def __init__(self):
        """Initialize the cleaner with comprehensive philosophical and religious keywords."""
        self.nlp = self._load_spacy_model()
        
        # Expanded philosophical and religious keywords
        self.philosophical_keywords = {
            # Core philosophical concepts
            'truth', 'wisdom', 'knowledge', 'understanding', 'enlightenment', 'consciousness',
            'reality', 'existence', 'being', 'essence', 'nature', 'eternal', 'infinite',
            'absolute', 'ultimate', 'supreme', 'divine', 'sacred', 'holy', 'blessed',
            
            # Religious and spiritual terms
            'god', 'lord', 'allah', 'brahman', 'buddha', 'christ', 'spirit', 'soul',
            'faith', 'belief', 'prayer', 'meditation', 'worship', 'devotion', 'reverence',
            'salvation', 'redemption', 'liberation', 'moksha', 'nirvana', 'heaven', 'paradise',
            
            # Moral and ethical concepts
            'love', 'compassion', 'mercy', 'forgiveness', 'justice', 'righteousness',
            'virtue', 'sin', 'evil', 'good', 'moral', 'ethical', 'dharma', 'karma',
            'peace', 'harmony', 'unity', 'oneness', 'transcendence',
            
            # Religious practices and concepts
            'scripture', 'revelation', 'prophecy', 'miracle', 'blessing', 'grace',
            'covenant', 'commandment', 'law', 'teaching', 'doctrine', 'theology',
            'creation', 'creator', 'manifest', 'divine', 'sacred', 'holy',
            
            # Mystical and philosophical terms
            'mystery', 'mystical', 'spiritual', 'transcendent', 'immanent', 'ineffable',
            'sublime', 'profound', 'deep', 'inner', 'higher', 'beyond', 'within',
            
            # Action and practice terms
            'meditate', 'pray', 'worship', 'contemplate', 'reflect', 'realize',
            'awaken', 'transform', 'purify', 'sanctify', 'bless', 'consecrate'
        }
        
        # Religious-specific protected terms
        self.religious_protected_terms = {
            # Hinduism/Buddhism
            'brahman', 'atman', 'dharma', 'karma', 'samsara', 'moksha', 'nirvana',
            'buddha', 'bodhisattva', 'enlightenment', 'meditation', 'mindfulness',
            'vedanta', 'upanishad', 'gita', 'sutra', 'tantra', 'yoga', 'prana',
            'jivanmukti', 'samadhi', 'satori', 'koan', 'zen',
            
            # Christianity
            'christ', 'jesus', 'gospel', 'salvation', 'redemption', 'trinity',
            'incarnation', 'resurrection', 'crucifixion', 'apostle', 'disciple',
            'parable', 'beatitude', 'grace', 'faith', 'hope', 'charity',
            
            # Islam
            'allah', 'prophet', 'quran', 'hadith', 'jihad', 'hajj', 'salah',
            'zakat', 'ramadan', 'ummah', 'imam', 'mosque', 'mecca', 'medina',
            
            # Judaism
            'torah', 'talmud', 'synagogue', 'rabbi', 'sabbath', 'kosher',
            'mitzvah', 'covenant', 'exodus', 'jerusalem', 'zion'
        }
        
        # Common genealogical and historical patterns to filter
        self.genealogy_patterns = [
            r'\b\w+ begat \w+',
            r'\b\w+ was the father of \w+',
            r'\b\w+ bore \w+',
            r'\bson of \w+\b',
            r'\bdaughter of \w+\b',
            r'\bking of \w+\b',
            r'\bruled \d+ years?',
            r'\bdied in the \w+ year',
            r'\bwent to \w+ and dwelt there',
            r'\bcame to pass in the days of',
        ]
        
        # Whitelist for important religious names and terms
        self.name_whitelist = {
            'God', 'Lord', 'Jesus', 'Christ', 'Buddha', 'Muhammad', 'Moses', 'Abraham',
            'Krishna', 'Rama', 'Shiva', 'Vishnu', 'Brahma', 'Allah', 'Yahweh',
            'Truth', 'Wisdom', 'Love', 'Justice', 'Peace', 'Light', 'Way',
            'Father', 'Son', 'Spirit', 'Holy', 'Sacred', 'Divine', 'Eternal',
            'Heaven', 'Earth', 'Creation', 'Creator', 'Almighty', 'Supreme'
        }

    def _load_spacy_model(self):
        """Load spaCy model with error handling."""
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model 'en_core_web_sm' not found. Some features may be limited.")
            return None

    def is_genealogical_text(self, text: str) -> bool:
        """Check if text is primarily genealogical information."""
        text_lower = text.lower()
        for pattern in self.genealogy_patterns:
            if re.search(pattern, text_lower):
                return True
        return False

    def contains_meaningful_content(self, text: str) -> bool:
        """Check if text contains meaningful philosophical or religious content."""
        text_lower = text.lower()
        
        # Check for philosophical keywords
        if any(keyword in text_lower for keyword in self.philosophical_keywords):
            return True
            
        # Check for protected religious terms
        if any(term in text_lower for term in self.religious_protected_terms):
            return True
            
        # Check for meaningful verbs that indicate teaching or action
        meaningful_verbs = {
            'teaches', 'reveals', 'declares', 'proclaims', 'speaks', 'says',
            'commands', 'instructs', 'guides', 'leads', 'shows', 'demonstrates',
            'manifests', 'embodies', 'represents', 'symbolizes', 'signifies',
            'means', 'indicates', 'suggests', 'implies', 'conveys'
        }
        
        if any(verb in text_lower for verb in meaningful_verbs):
            return True
            
        return False

    def calculate_capital_ratio(self, text: str) -> float:
        """Calculate ratio of capitalized words, excluding whitelisted terms."""
        try:
            # Handle None or non-string inputs
            if not isinstance(text, str) or not text or not text.strip():
                return 0.0
            
            words = word_tokenize(text)
            if len(words) < 3:
                return 0.0
            
            alpha_words = [w for w in words if w.isalpha() and len(w) > 2]
            if not alpha_words:
                return 0.0
            
            # Don't count first word or whitelisted words
            capitalized = [
                w for i, w in enumerate(alpha_words)
                if w[0].isupper() and w not in self.name_whitelist and i != 0
            ]
            
            return len(capitalized) / len(alpha_words)
        except Exception as e:
            # If tokenization fails, return safe default
            print(f"Warning: Error tokenizing text: {e}")
            return 0.0

    def is_mostly_names(self, text: str) -> bool:
        """Check if text is mostly proper names using NER."""
        if not self.nlp:
            return False
            
        doc = self.nlp(text)
        if len(doc) == 0:
            return False
            
        # Count tokens that are part of named entities (PERSON, GPE, ORG)
        entity_tokens = set()
        for ent in doc.ents:
            if ent.label_ in {"PERSON", "GPE", "ORG"}:
                # Don't count religious figures or important places
                if any(protected in ent.text.lower() for protected in self.religious_protected_terms):
                    continue
                if ent.text in self.name_whitelist:
                    continue
                entity_tokens.update(range(ent.start, ent.end))
        
        # Also count standalone capitalized words that aren't whitelisted
        for i, token in enumerate(doc):
            if (token.text[0].isupper() and 
                token.text not in self.name_whitelist and 
                i != 0 and 
                token.is_alpha and 
                len(token.text) > 2):
                entity_tokens.add(i)
        
        ratio = len(entity_tokens) / len(doc)
        return ratio > 0.6  # More lenient threshold

    def has_meaningful_length(self, text: str) -> bool:
        """Check if text has meaningful length and complexity."""
        words = text.split()
        
        # Too short
        if len(words) < 4:
            return False
            
        # Check for very repetitive text
        unique_words = set(w.lower() for w in words if w.isalpha())
        if len(unique_words) < len(words) * 0.3:  # Less than 30% unique words
            return False
            
        return True

    def is_valuable_short_text(self, text: str) -> bool:
        """Check if short text is still valuable (quotes, key teachings)."""
        text_lower = text.lower()
        
        # Important short phrases
        valuable_patterns = [
            r'\bi am\b.*\bthat\b',  # "I am that I am" type statements
            r'\bthou art\b',        # "Thou art that"
            r'\bknow.*truth\b',     # Know the truth
            r'\blive.*peace\b',     # Live in peace
            r'\blove.*neighbor\b',  # Love thy neighbor
            r'\bseek.*find\b',      # Seek and find
            r'\bask.*given\b',      # Ask and it shall be given
            r'\bblessed.*\b',       # Blessed are...
            r'\breturn.*lord\b',    # Return to the Lord
            r'\bpure.*heart\b',     # Pure in heart
        ]
        
        for pattern in valuable_patterns:
            if re.search(pattern, text_lower):
                return True
                
        return False

    def should_keep_sentence(self, text: str, source: str = "") -> Dict[str, any]:
        """
        Comprehensive decision on whether to keep a sentence.
        Returns dict with 'keep' boolean and 'reason' string.
        """
        try:
            # Handle None or non-string inputs
            if not isinstance(text, str):
                return {'keep': False, 'reason': 'invalid_input'}
                
            text = text.strip()
            
            # Always keep if empty or very basic
            if not text or len(text) < 3:
                return {'keep': False, 'reason': 'empty_or_too_short'}
            
            # Check for genealogical content first
            if self.is_genealogical_text(text):
                return {'keep': False, 'reason': 'genealogical_content'}
            
            # Check length requirements
            if not self.has_meaningful_length(text):
                if self.is_valuable_short_text(text):
                    return {'keep': True, 'reason': 'valuable_short_text'}
                else:
                    return {'keep': False, 'reason': 'too_short_not_valuable'}
            
            # Check for meaningful content
            if self.contains_meaningful_content(text):
                # Even if high capital ratio, keep if meaningful
                return {'keep': True, 'reason': 'contains_meaningful_content'}
            
            # Check capital ratio (more lenient for meaningful content)
            capital_ratio = self.calculate_capital_ratio(text)
            if capital_ratio > 0.5:  # Very high threshold
                if not self.is_mostly_names(text):
                    return {'keep': True, 'reason': 'high_capitals_but_not_names'}
                else:
                    return {'keep': False, 'reason': 'mostly_names_high_capitals'}
            
            # Check if mostly names
            if self.is_mostly_names(text):
                return {'keep': False, 'reason': 'mostly_names'}
            
            # Default: keep if we've gotten this far
            return {'keep': True, 'reason': 'passed_all_filters'}
            
        except Exception as e:
            # If any error occurs, default to keeping the sentence
            print(f"Warning: Error processing sentence: {e}")
            return {'keep': True, 'reason': 'error_default_keep'}

    def clean_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """Clean the entire dataframe and add decision metadata."""
        
        print(f"Starting with {len(df)} rows")
        
        # Apply cleaning logic
        decisions = df[text_column].apply(
            lambda x: self.should_keep_sentence(x, df.get('source', [''])[0] if len(df) > 0 else '')
        )
        
        # Add decision columns
        df['keep'] = decisions.apply(lambda x: x['keep'])
        df['filter_reason'] = decisions.apply(lambda x: x['reason'])
        
        # Statistics
        total_original = len(df)
        total_kept = df['keep'].sum()
        total_dropped = total_original - total_kept
        
        print(f"\nCleaning Results:")
        print(f"Original sentences: {total_original}")
        print(f"Kept: {total_kept} ({total_kept/total_original*100:.1f}%)")
        print(f"Dropped: {total_dropped} ({total_dropped/total_original*100:.1f}%)")
        
        print(f"\nReasons for dropping:")
        drop_reasons = df[~df['keep']]['filter_reason'].value_counts()
        for reason, count in drop_reasons.items():
            print(f"  {reason}: {count} ({count/total_original*100:.1f}%)")
        
        print(f"\nReasons for keeping:")
        keep_reasons = df[df['keep']]['filter_reason'].value_counts()
        for reason, count in keep_reasons.items():
            print(f"  {reason}: {count} ({count/total_original*100:.1f}%)")
        
        return df

def main():
    """Main function to clean the CSV file."""
    
    # Initialize cleaner
    cleaner = AdvancedReligiousTextCleaner()
    
    # Load main data with encoding detection
    print("Loading religion_data.csv...")
    df = None
    
    # Try different encodings
    encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
    
    for encoding in encodings_to_try:
        try:
            print(f"Trying encoding: {encoding}")
            df = pd.read_csv("religion_data.csv", encoding=encoding)
            print(f"Successfully loaded with {encoding} encoding")
            break
        except (UnicodeDecodeError, FileNotFoundError) as e:
            if encoding == encodings_to_try[-1]:  # Last encoding attempt
                if isinstance(e, FileNotFoundError):
                    print("Error: religion_data.csv not found!")
                    return
                else:
                    print(f"Failed to load with any encoding. Last error: {e}")
                    return
            else:
                continue
    
    if df is None:
        print("Could not load the CSV file")
        return
    
    print(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
    
    # Load previously dropped sentences to re-evaluate
    previously_dropped = None
    for encoding in encodings_to_try:
        try:
            previously_dropped = pd.read_csv("dropped_sentences.csv", encoding=encoding)
            print(f"Loaded {len(previously_dropped)} previously dropped sentences for re-evaluation")
            break
        except (FileNotFoundError, UnicodeDecodeError):
            continue
    
    if previously_dropped is None:
        print("Note: dropped_sentences.csv not found - will only process main data")
    
    # Clean the main dataframe
    df_with_decisions = cleaner.clean_dataframe(df)
    
    # Create clean version (only kept sentences)
    df_clean = df_with_decisions[df_with_decisions['keep']].copy()
    df_clean = df_clean.drop(columns=['keep', 'filter_reason'])
    
    # Create dropped version for analysis
    df_dropped = df_with_decisions[~df_with_decisions['keep']].copy()
    
    # Save main results (with utf-8 encoding)
    df_clean.to_csv("religion_data_clean.csv", index=False, encoding='utf-8')
    print(f"\nSaved {len(df_clean)} clean sentences to 'religion_data_clean.csv'")
    
    # Save full version with decisions for analysis
    df_with_decisions.to_csv("religion_data_with_decisions.csv", index=False, encoding='utf-8')
    print(f"Saved full analysis to 'religion_data_with_decisions.csv'")
    
    # Save dropped sentences for review
    if len(df_dropped) > 0:
        df_dropped[['text', 'filter_reason'] + [col for col in df.columns if col not in ['text', 'keep', 'filter_reason']]].to_csv("dropped_sentences_detailed.csv", index=False, encoding='utf-8')
        print(f"Saved {len(df_dropped)} dropped sentences to 'dropped_sentences_detailed.csv'")
    
    # Process previously dropped sentences if available
    if previously_dropped is not None:
        print(f"\n--- Re-evaluating Previously Dropped Sentences ---")
        
        # Apply new cleaning logic to previously dropped sentences
        prev_decisions = previously_dropped['text'].apply(
            lambda x: cleaner.should_keep_sentence(x, "")
        )
        
        # Add decision columns
        previously_dropped['keep'] = prev_decisions.apply(lambda x: x['keep'])
        previously_dropped['new_filter_reason'] = prev_decisions.apply(lambda x: x['reason'])
        
        # Split into kept and dropped
        prev_kept = previously_dropped[previously_dropped['keep']].copy()
        prev_still_dropped = previously_dropped[~previously_dropped['keep']].copy()
        
        # Statistics for previously dropped sentences
        total_prev = len(previously_dropped)
        rescued = len(prev_kept)
        still_dropped = len(prev_still_dropped)
        
        print(f"Previously dropped sentences: {total_prev}")
        print(f"Now rescued (kept): {rescued} ({rescued/total_prev*100:.1f}%)")
        print(f"Still dropped: {still_dropped} ({still_dropped/total_prev*100:.1f}%)")
        
        # Show what we rescued
        if rescued > 0:
            print(f"\nReasons for rescuing previously dropped sentences:")
            rescue_reasons = prev_kept['new_filter_reason'].value_counts()
            for reason, count in rescue_reasons.items():
                print(f"  {reason}: {count}")
        
        # Show what's still being dropped
        if still_dropped > 0:
            print(f"\nReasons for still dropping sentences:")
            still_drop_reasons = prev_still_dropped['new_filter_reason'].value_counts()
            for reason, count in still_drop_reasons.items():
                print(f"  {reason}: {count}")
        
        # Save the rescued sentences (these were previously dropped but now kept)
        if rescued > 0:
            # Keep original columns plus new decision info
            kept_cols = ['text', 'drop_reason', 'school', 'new_filter_reason']
            available_cols = [col for col in kept_cols if col in prev_kept.columns]
            prev_kept[available_cols].to_csv("kept.csv", index=False, encoding='utf-8')
            print(f"\nSaved {rescued} rescued sentences to 'kept.csv'")
        else:
            # Create empty kept.csv
            pd.DataFrame(columns=['text', 'drop_reason', 'school', 'new_filter_reason']).to_csv("kept.csv", index=False, encoding='utf-8')
            print(f"\nNo sentences were rescued - created empty 'kept.csv'")
        
        # Save the still dropped sentences
        if still_dropped > 0:
            # Keep original columns plus new decision info
            dropped_cols = ['text', 'drop_reason', 'school', 'new_filter_reason']
            available_cols = [col for col in dropped_cols if col in prev_still_dropped.columns]
            prev_still_dropped[available_cols].to_csv("dropped.csv", index=False, encoding='utf-8')
            print(f"Saved {still_dropped} still-dropped sentences to 'dropped.csv'")
        else:
            # Create empty dropped.csv
            pd.DataFrame(columns=['text', 'drop_reason', 'school', 'new_filter_reason']).to_csv("dropped.csv", index=False, encoding='utf-8')
            print(f"All previously dropped sentences were rescued - created empty 'dropped.csv'")
        
        # Save full re-evaluation results
        all_cols = ['text', 'drop_reason', 'school', 'keep', 'new_filter_reason']
        available_cols = [col for col in all_cols if col in previously_dropped.columns]
        previously_dropped[available_cols].to_csv("previously_dropped_reevaluated.csv", index=False, encoding='utf-8')
        print(f"Saved full re-evaluation to 'previously_dropped_reevaluated.csv'")

if __name__ == "__main__":
    main()