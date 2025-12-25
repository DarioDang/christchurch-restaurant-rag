"""
Enhanced Evaluation Worker for Restaurant RAG System
Runs continuous evaluation of Phoenix traces with Shadow Geographic evaluation
"""

import os
import sys
import time
import functools
import warnings
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Tuple

import pytz

# Force print flushing
print = functools.partial(print, flush=True)
sys.stdout.flush()

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Force Phoenix endpoint before importing
endpoint_env = os.getenv("PHOENIX_CLIENT_ENDPOINT", "http://localhost:6006")
os.environ["PHOENIX_CLIENT_ENDPOINT"] = endpoint_env
print(f"[Worker] PHOENIX_CLIENT_ENDPOINT ENV = {endpoint_env}")

# Phoenix imports
import phoenix as px
from phoenix.session.evaluation import (
    get_retrieved_documents,
    get_qa_with_reference,
)
from phoenix.evals import (
    OpenAIModel,
    RelevanceEvaluator,
    HallucinationEvaluator,
    QAEvaluator,
    run_evals,
)
from phoenix.trace import DocumentEvaluations, SpanEvaluations

# Local imports
from .evaluators import (
    LocationAccuracyEvaluator,
    MetadataEnrichmentEvaluator,
    ShadowGeographicEvaluator,
)
from .checkpoint import load_checkpoint, save_checkpoint


# ============================================
# CONFIGURATION
# ============================================

PROJECT_NAME = "restaurant-rag-project"
POLL_INTERVAL_SECONDS = 60
OPENAI_MODEL = "gpt-4o-mini"
NZ_TZ = pytz.timezone("Pacific/Auckland")


# ============================================
# UTILITY FUNCTIONS
# ============================================

def to_nz(dt_utc: datetime) -> str:
    """
    Convert UTC datetime to NZ local time string
    
    Args:
        dt_utc: UTC datetime
    
    Returns:
        Formatted NZ time string
    """
    return dt_utc.astimezone(NZ_TZ).strftime("%Y-%m-%d %H:%M:%S NZDT")


def nz(dt_utc: datetime) -> str:
    """
    Convert UTC datetime to NZ time with timezone
    
    Args:
        dt_utc: UTC datetime
    
    Returns:
        Formatted NZ time with timezone
    """
    local = dt_utc.astimezone(NZ_TZ)
    return local.strftime("%Y-%m-%d %H:%M:%S %Z")


def verify_phoenix_connection(endpoint: str) -> bool:
    """
    Verify Phoenix server is reachable
    
    Args:
        endpoint: Phoenix server URL
    
    Returns:
        True if reachable, False otherwise
    """
    try:
        health_check = requests.get(f"{endpoint}/healthz", timeout=5)
        if health_check.status_code == 200:
            print("[Worker] ✓ Phoenix is reachable")
            return True
        else:
            print(f"[Worker] ⚠️  Phoenix returned status {health_check.status_code}")
            return False
    except Exception as e:
        print(f"[Worker] ⚠️  WARNING: Cannot reach Phoenix: {e}")
        print("[Worker] Make sure Phoenix is running: phoenix serve")
        return False


def create_phoenix_client(endpoint: str) -> Optional[px.Client]:
    """
    Create Phoenix client with error handling
    
    Args:
        endpoint: Phoenix server URL
    
    Returns:
        Phoenix client or None if failed
    """
    try:
        client = px.Client(endpoint=endpoint)
        print("[Worker] ✓ Phoenix client created")
        return client
    except Exception as e:
        print(f"[Worker] ERROR creating Phoenix client: {e}")
        return None


def create_evaluators(eval_model: OpenAIModel) -> Tuple:
    """
    Create all evaluator instances
    
    Args:
        eval_model: OpenAI model for evaluations
    
    Returns:
        Tuple of (standard_evaluators, custom_evaluators)
    """
    # Standard Phoenix evaluators
    relevance_evaluator = RelevanceEvaluator(eval_model)
    hallucination_evaluator = HallucinationEvaluator(eval_model)
    qa_evaluator = QAEvaluator(eval_model)
    
    # Custom evaluators
    shadow_geo_evaluator = ShadowGeographicEvaluator(eval_model)
    location_evaluator = LocationAccuracyEvaluator(eval_model)
    metadata_evaluator = MetadataEnrichmentEvaluator(eval_model)
    
    print("[Worker] ✓ Created standard + custom evaluators")
    
    return (
        (relevance_evaluator, hallucination_evaluator, qa_evaluator),
        (shadow_geo_evaluator, location_evaluator, metadata_evaluator)
    )


# ============================================
# DATA FETCHING
# ============================================

def fetch_evaluation_data(
    client: px.Client,
    start_time: datetime,
    end_time: datetime
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Fetch retrieved documents and QA spans from Phoenix
    
    Args:
        client: Phoenix client
        start_time: Window start (UTC)
        end_time: Window end (UTC)
    
    Returns:
        Tuple of (retrieved_docs_df, qa_df)
    """
    retrieved_docs_df = None
    qa_df = None
    
    # Fetch retrieved documents
    try:
        retrieved_docs_df = get_retrieved_documents(
            client,
            project_name=PROJECT_NAME,
            start_time=start_time,
            end_time=end_time,
        )
    except Exception as e:
        print(f"[Worker] Error fetching retrieved documents: {e}")
    
    # Fetch QA spans
    try:
        qa_df = get_qa_with_reference(
            client,
            project_name=PROJECT_NAME,
            start_time=start_time,
            end_time=end_time,
        )
    except Exception as e:
        print(f"[Worker] Error fetching QA data: {e}")
    
    return retrieved_docs_df, qa_df


def retry_qa_fetch(
    client: px.Client,
    start_time: datetime,
    retry_wait: int = 20
) -> Optional[pd.DataFrame]:
    """
    Retry fetching QA spans after a delay
    
    Args:
        client: Phoenix client
        start_time: Window start time
        retry_wait: Seconds to wait before retry
    
    Returns:
        QA DataFrame or None
    """
    print(f"[Worker] Waiting {retry_wait} seconds before QA retry...")
    time.sleep(retry_wait)
    
    retry_end_time = datetime.now(timezone.utc) + timedelta(seconds=5)
    
    try:
        qa_df = get_qa_with_reference(
            client,
            project_name=PROJECT_NAME,
            start_time=start_time,
            end_time=retry_end_time,
        )
        
        qa_count = 0 if qa_df is None else len(qa_df)
        print(f"[Worker] After retry: Found {qa_count} QA rows")
        return qa_df
        
    except Exception as e:
        print(f"[Worker] Error during QA retry: {e}")
        return None


# ============================================
# EVALUATION RUNNING
# ============================================

def run_standard_evaluations(
    retrieved_docs_df: Optional[pd.DataFrame],
    qa_df: Optional[pd.DataFrame],
    evaluators: Tuple
) -> Tuple:
    """
    Run standard Phoenix evaluations
    
    Args:
        retrieved_docs_df: Retrieved documents
        qa_df: QA spans
        evaluators: Tuple of (relevance, hallucination, qa)
    
    Returns:
        Tuple of (relevance_df, hallucination_df, qa_correctness_df)
    """
    relevance_evaluator, hallucination_evaluator, qa_evaluator = evaluators
    
    retrieved_relevance_df = None
    hallucination_eval_df = None
    qa_eval_df = None
    
    # Run retrieval relevance evaluation
    if retrieved_docs_df is not None and len(retrieved_docs_df) > 0:
        print("[Worker] Running Retrieval Relevance eval...")
        try:
            retrieved_relevance_df = run_evals(
                dataframe=retrieved_docs_df,
                evaluators=[relevance_evaluator],
                provide_explanation=True,
                concurrency=20,
            )[0]
            
            # Print summary
            if retrieved_relevance_df is not None and len(retrieved_relevance_df) > 0:
                label_col = [c for c in retrieved_relevance_df.columns if 'label' in c.lower()]
                if label_col:
                    score_summary = retrieved_relevance_df[label_col[0]].value_counts()
                    print(f"[Worker] Relevance scores:")
                    for label, count in score_summary.items():
                        print(f"  - {label}: {count}")
                        
        except Exception as e:
            print(f"[Worker] Warning during retrieval eval: {e}")
            import traceback
            traceback.print_exc()
    
    # Run hallucination and QA evaluations
    if qa_df is not None and len(qa_df) > 0:
        print("[Worker] Running Hallucination + QA Correctness evals...")
        
        # Check for expected sections in reference
        if 'reference' in qa_df.columns and len(qa_df) > 0:
            sample_ref = str(qa_df.iloc[0]['reference'])
            if '[SHADOW_EVALUATION_DATA]' in sample_ref:
                print("[Worker] ✓ Reference contains [SHADOW_EVALUATION_DATA]")
            elif '[RESTAURANT INFO]' in sample_ref:
                print("[Worker] ✓ Reference contains [RESTAURANT INFO]")
            else:
                print("[Worker] ⚠️  Reference missing expected sections")
        
        try:
            eval_results = run_evals(
                dataframe=qa_df,
                evaluators=[hallucination_evaluator, qa_evaluator],
                provide_explanation=True,
                concurrency=20,
            )
            
            hallucination_eval_df = eval_results[0]
            qa_eval_df = eval_results[1]
            
            # Print summaries
            if hallucination_eval_df is not None and len(hallucination_eval_df) > 0:
                label_col = [c for c in hallucination_eval_df.columns if 'label' in c.lower()]
                if label_col:
                    score_summary = hallucination_eval_df[label_col[0]].value_counts()
                    print(f"[Worker] Hallucination scores:")
                    for label, count in score_summary.items():
                        print(f"  - {label}: {count}")
            
            if qa_eval_df is not None and len(qa_eval_df) > 0:
                label_col = [c for c in qa_eval_df.columns if 'label' in c.lower()]
                if label_col:
                    score_summary = qa_eval_df[label_col[0]].value_counts()
                    print(f"[Worker] QA Correctness scores:")
                    for label, count in score_summary.items():
                        print(f"  - {label}: {count}")
                        
        except Exception as e:
            print(f"[Worker] Warning during QA eval: {e}")
            import traceback
            traceback.print_exc()
    
    return retrieved_relevance_df, hallucination_eval_df, qa_eval_df


def run_custom_evaluations(
    qa_df: Optional[pd.DataFrame],
    evaluators: Tuple
) -> Tuple:
    """
    Run custom evaluations (Shadow Geographic, Location, Metadata)
    
    Args:
        qa_df: QA spans DataFrame
        evaluators: Tuple of (shadow_geo, location, metadata)
    
    Returns:
        Tuple of (shadow_geo_df, location_df, metadata_df)
    """
    shadow_geo_evaluator, location_evaluator, metadata_evaluator = evaluators
    
    shadow_geo_eval_df = None
    location_eval_df = None
    metadata_eval_df = None
    
    if qa_df is None or len(qa_df) == 0:
        return None, None, None
    
    print("[Worker] Running custom evaluations...")
    
    try:
        shadow_geo_results = []
        location_results = []
        metadata_results = []
        
        for idx, row in qa_df.iterrows():
            # Extract fields
            input_text = str(row.get('input', ''))
            output_text = str(row.get('output', ''))
            reference_text = str(row.get('reference', ''))
            span_id = row.get('context.span_id', f'span_{idx}')
            
            # Shadow Geographic evaluation
            try:
                shadow_result = shadow_geo_evaluator.evaluate(
                    input_text, output_text, reference_text
                )
                shadow_result['context.span_id'] = span_id
                shadow_geo_results.append(shadow_result)
            except Exception as e:
                print(f"[Worker] Error in shadow geo eval for span {span_id}: {e}")
            
            # Location accuracy evaluation
            try:
                loc_result = location_evaluator.evaluate(
                    input_text, output_text, reference_text
                )
                loc_result['context.span_id'] = span_id
                location_results.append(loc_result)
            except Exception as e:
                print(f"[Worker] Error in location eval for span {span_id}: {e}")
            
            # Metadata enrichment evaluation
            try:
                meta_result = metadata_evaluator.evaluate(
                    input_text, output_text, reference_text
                )
                meta_result['context.span_id'] = span_id
                metadata_results.append(meta_result)
            except Exception as e:
                print(f"[Worker] Error in metadata eval for span {span_id}: {e}")
        
        # Convert to DataFrames and print summaries
        if shadow_geo_results:
            shadow_geo_eval_df = pd.DataFrame(shadow_geo_results)
            shadow_summary = shadow_geo_eval_df['label'].value_counts()
            print(f"[Worker] 🌍 Shadow Geographic Accuracy scores:")
            for label, count in shadow_summary.items():
                print(f"  - {label}: {count}")
        
        if location_results:
            location_eval_df = pd.DataFrame(location_results)
            location_summary = location_eval_df['label'].value_counts()
            print(f"[Worker] 📍 Location Accuracy scores:")
            for label, count in location_summary.items():
                print(f"  - {label}: {count}")
        
        if metadata_results:
            metadata_eval_df = pd.DataFrame(metadata_results)
            metadata_summary = metadata_eval_df['label'].value_counts()
            print(f"[Worker] 📊 Metadata Enrichment scores:")
            for label, count in metadata_summary.items():
                print(f"  - {label}: {count}")
        
    except Exception as e:
        print(f"[Worker] Warning during custom evals: {e}")
        import traceback
        traceback.print_exc()
    
    return shadow_geo_eval_df, location_eval_df, metadata_eval_df


def log_evaluations_to_phoenix(
    client: px.Client,
    retrieved_relevance_df: Optional[pd.DataFrame],
    hallucination_eval_df: Optional[pd.DataFrame],
    qa_eval_df: Optional[pd.DataFrame],
    shadow_geo_eval_df: Optional[pd.DataFrame],
    location_eval_df: Optional[pd.DataFrame],
    metadata_eval_df: Optional[pd.DataFrame]
) -> None:
    """
    Log all evaluation results to Phoenix
    
    Args:
        client: Phoenix client
        *_df: Evaluation result DataFrames
    """
    evals_to_log = []
    
    # Standard evaluations
    if hallucination_eval_df is not None and len(hallucination_eval_df) > 0:
        evals_to_log.append(
            SpanEvaluations(
                eval_name="Hallucination",
                dataframe=hallucination_eval_df,
            )
        )
    
    if qa_eval_df is not None and len(qa_eval_df) > 0:
        evals_to_log.append(
            SpanEvaluations(
                eval_name="QA Correctness",
                dataframe=qa_eval_df,
            )
        )
    
    if retrieved_relevance_df is not None and len(retrieved_relevance_df) > 0:
        evals_to_log.append(
            DocumentEvaluations(
                eval_name="Retrieval Relevance",
                dataframe=retrieved_relevance_df,
            )
        )
    
    # Custom evaluations
    if shadow_geo_eval_df is not None and len(shadow_geo_eval_df) > 0:
        evals_to_log.append(
            SpanEvaluations(
                eval_name="Shadow Geographic Accuracy",
                dataframe=shadow_geo_eval_df,
            )
        )
        print(f"[Worker] ✓ Will log {len(shadow_geo_eval_df)} Shadow Geographic evaluations")
    
    if location_eval_df is not None and len(location_eval_df) > 0:
        evals_to_log.append(
            SpanEvaluations(
                eval_name="Location Accuracy",
                dataframe=location_eval_df,
            )
        )
        print(f"[Worker] ✓ Will log {len(location_eval_df)} Location Accuracy evaluations")
    
    if metadata_eval_df is not None and len(metadata_eval_df) > 0:
        evals_to_log.append(
            SpanEvaluations(
                eval_name="Metadata Enrichment",
                dataframe=metadata_eval_df,
            )
        )
        print(f"[Worker] ✓ Will log {len(metadata_eval_df)} Metadata Enrichment evaluations")
    
    # Log to Phoenix
    if evals_to_log:
        print(f"[Worker] Logging {len(evals_to_log)} evaluation tables to Phoenix...")
        try:
            client.log_evaluations(*evals_to_log)
            print("[Worker] ✓ Logged all evaluations successfully")
        except Exception as log_err:
            print(f"[Worker] ERROR logging evaluations: {log_err}")
            import traceback
            traceback.print_exc()
    else:
        print("[Worker] Nothing to log (no eval outputs)")


# ============================================
# MAIN EVALUATION CYCLE
# ============================================

def run_eval_cycle(client: px.Client, start_time: datetime) -> datetime:
    """
    Run one complete evaluation cycle
    
    Args:
        client: Phoenix client
        start_time: Start of evaluation window (UTC)
    
    Returns:
        Updated checkpoint time (UTC)
    """
    # Capture window before waiting
    initial_end_time = datetime.now(timezone.utc)
    
    # Wait for in-flight traces to complete
    wait_time = 40
    print(f"[Worker] Waiting {wait_time}s for in-flight traces to complete...")
    time.sleep(wait_time)
    
    # Extend window after waiting
    actual_end_time = datetime.now(timezone.utc)
    
    # Add buffers
    forward_buffer = timedelta(seconds=20)
    extended_end_time = actual_end_time + forward_buffer
    
    lookback_buffer = timedelta(seconds=20)
    adjusted_start = start_time - lookback_buffer
    
    # Print evaluation window
    print(f"\n[Worker] Evaluating window:")
    print(f"[Worker]   Start: {nz(adjusted_start)}")
    print(f"[Worker]   End:   {nz(extended_end_time)}")
    print(f"[Worker]   Total window: {(extended_end_time - adjusted_start).total_seconds():.0f} seconds")
    
    # Fetch data
    retrieved_docs_df, qa_df = fetch_evaluation_data(
        client, adjusted_start, extended_end_time
    )
    
    # Check if we found any spans
    if (retrieved_docs_df is None or len(retrieved_docs_df) == 0) and \
       (qa_df is None or len(qa_df) == 0):
        print("[Worker] No new spans found in this window")
        return actual_end_time
    
    print(
        f"[Worker] Found "
        f"{0 if retrieved_docs_df is None else len(retrieved_docs_df)} retrieved-doc rows, "
        f"{0 if qa_df is None else len(qa_df)} QA rows"
    )
    
    # Retry QA fetch if we have docs but no QA
    if (retrieved_docs_df is not None and len(retrieved_docs_df) > 0) and \
       (qa_df is None or len(qa_df) == 0):
        qa_df = retry_qa_fetch(client, adjusted_start)
        actual_end_time = datetime.now(timezone.utc)
    
    # Create evaluators
    try:
        eval_model = OpenAIModel(
            model=OPENAI_MODEL,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        standard_evaluators, custom_evaluators = create_evaluators(eval_model)
        
    except Exception as e:
        print(f"[Worker] ERROR creating evaluators: {e}")
        print("[Worker] Make sure OPENAI_API_KEY is set")
        import traceback
        traceback.print_exc()
        return actual_end_time
    
    # Run standard evaluations
    retrieved_relevance_df, hallucination_eval_df, qa_eval_df = run_standard_evaluations(
        retrieved_docs_df, qa_df, standard_evaluators
    )
    
    # Run custom evaluations
    shadow_geo_eval_df, location_eval_df, metadata_eval_df = run_custom_evaluations(
        qa_df, custom_evaluators
    )
    
    # Log all evaluations to Phoenix
    log_evaluations_to_phoenix(
        client,
        retrieved_relevance_df,
        hallucination_eval_df,
        qa_eval_df,
        shadow_geo_eval_df,
        location_eval_df,
        metadata_eval_df
    )
    
    return actual_end_time


# ============================================
# MAIN WORKER LOOP
# ============================================

def main():
    """Main worker loop - runs continuous evaluation"""
    
    print("=" * 70)
    print("[Worker] Starting Enhanced Evaluation Worker")
    print("[Worker] Features:")
    print("[Worker]   ✓ Standard evaluators (Relevance, Hallucination, QA)")
    print("[Worker]   ✓ Shadow Geographic Accuracy (coordinate validation)")
    print("[Worker]   ✓ Location Accuracy (distance sorting)")
    print("[Worker]   ✓ Metadata Enrichment (address, hours, etc.)")
    print("=" * 70)
    
    # Get Phoenix endpoint
    endpoint = os.getenv("PHOENIX_CLIENT_ENDPOINT", "http://localhost:6006")
    print(f"[Worker] Using Phoenix endpoint: {endpoint}")
    
    # Verify connection
    verify_phoenix_connection(endpoint)
    
    # Create client
    client = create_phoenix_client(endpoint)
    if client is None:
        print("[Worker] Exiting...")
        return
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("[Worker] ⚠️  WARNING: OPENAI_API_KEY not set")
        print("[Worker] Set it with: export OPENAI_API_KEY='sk-...'")
        print("[Worker] Evaluations will fail without it")
    
    # Load checkpoint
    last_time = load_checkpoint()
    print(f"[Worker] Initial checkpoint (NZ): {nz(last_time)}")
    print(f"[Worker] Checking every {POLL_INTERVAL_SECONDS} seconds")
    print("=" * 70)
    
    cycle_count = 0
    
    # Main loop
    while True:
        try:
            cycle_count += 1
            print(f"\n{'=' * 70}")
            print(f"[Worker] === CYCLE {cycle_count} ===")
            print(f"{'=' * 70}")
            
            last_time = run_eval_cycle(client, last_time)
            save_checkpoint(last_time)
            
            print(f"[Worker] Checkpoint saved (NZ): {nz(last_time)}")
            
        except KeyboardInterrupt:
            print("\n[Worker] Received interrupt signal. Shutting down...")
            save_checkpoint(last_time)
            print("[Worker] Final checkpoint saved")
            break
        
        except Exception as e:
            print(f"[Worker] ERROR during eval cycle: {e}")
            import traceback
            traceback.print_exc()
            print(f"[Worker] Will retry on next cycle...")
        
        # Sleep until next cycle
        print(f"\n[Worker] Sleeping for {POLL_INTERVAL_SECONDS} seconds...")
        next_check_utc = datetime.now(timezone.utc) + timedelta(seconds=POLL_INTERVAL_SECONDS)
        print(f"[Worker] Next check at (NZ): {nz(next_check_utc)}")
        time.sleep(POLL_INTERVAL_SECONDS)


# ============================================
# ENTRY POINT
# ============================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("Test mode not implemented yet")
        sys.exit(0)
    else:
        try:
            main()
        except KeyboardInterrupt:
            print("\n[Worker] Interrupted. Exiting...")
            sys.exit(0)