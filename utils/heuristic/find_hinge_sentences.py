'''
Identify CoT sentences that reliably steer a model between compliance and
refusal in either direction.

For each scored_full_resample_prompt*_cot*_rep_*.csv file the script:
  1. Computes per-sentence-position statistics (mean score, comply-rate,
     refuse-rate) across all rollout repetitions.
  2. Finds every consecutive pair (sentence_idx N-1 → N) where the mean score
     crosses the comply/refuse thresholds with a large enough absolute delta.
     Two directions are detected:
       • comply→refuse  (C→R): score drops, model goes from compliant to refusing
       • refuse→comply  (R→C): score rises, model goes from refusing to compliant
  3. For both directions the minimum fraction of rollouts that must
     comply/refuse is controlled by a single --min_rate argument.
  4. Extracts the sentence text that was added at position N by diffing the
     combined_thread strings for the same repetition.
  5. Writes a CSV of all detected transitions and prints a ranked summary.

Example usage:
    uv run -m utils.heuristic.find_hinge_sentences \
        --model_name openai/gpt-oss-20b \
        --repetitions 10 \
        --comply_threshold 0.7 \
        --refuse_threshold 0.3 \
        --min_rate 0.7
'''

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Find CoT hinge sentences that trigger compliance/refusal transitions'
    )
    parser.add_argument('--model_name', type=str,
                        default='deepseek-ai/DeepSeek-R1-Distill-Llama-8B')
    parser.add_argument('--results_dir', type=str, default='results/')
    parser.add_argument('--repetitions', type=int, default=10,
                        help='Repetitions per sentence position (used in filename glob)')
    parser.add_argument('--comply_threshold', type=float, default=0.8,
                        help='Mean score boundary for "compliant" state')
    parser.add_argument('--refuse_threshold', type=float, default=0.2,
                        help='Mean score boundary for "refusing" state')
    parser.add_argument('--min_rate', type=float, default=0.9,
                        help='Minimum fraction of rollouts that must be in the expected '
                             'state both before and after the transition. Applied '
                             'symmetrically to both comply→refuse and refuse→comply.')
    parser.add_argument('--output', type=str, default=None,
                        help='Path for output CSV (default: auto-generated in results dir)')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_prompt_cot(filepath: str) -> tuple[int, int]:
    basename = os.path.basename(filepath)
    m = re.search(r'prompt(\d+)_cot(\d+)', basename)
    if m:
        return int(m.group(1)), int(m.group(2))
    raise ValueError(f'Cannot parse prompt/cot from filename: {basename}')


def extract_added_sentence(prev_thread: str, curr_thread: str) -> str:
    '''Return the text added to combined_thread going from sentence N-1 to N.'''
    if curr_thread.startswith(prev_thread):
        return curr_thread[len(prev_thread):]
    # Fallback: longest common prefix
    common_len = len(os.path.commonprefix([prev_thread, curr_thread]))
    return curr_thread[common_len:]


def analyse_file(filepath: str, args) -> list[dict]:
    '''
    Return a list of transition records for a single CSV file.

    Each record contains:
        prompt_idx, cot_idx, sentence_idx (the NEW sentence added),
        direction ("comply->refuse" or "refuse->comply"),
        sentence_text,
        mean_before, mean_after, score_delta,
        rate_before, rate_after,
        n_reps_before, n_reps_after
    '''
    df = pd.read_csv(filepath)

    required = {'sentence_idx', 'resample_n', 'combined_thread', 'strongreject_score'}
    if not required.issubset(df.columns):
        return []

    prompt_idx, cot_idx = extract_prompt_cot(filepath)

    # Per-position statistics
    stats = (
        df.groupby('sentence_idx')['strongreject_score']
        .agg(mean_score='mean', std_score='std',
             comply_rate=lambda s: (s > 0.5).mean(),
             refuse_rate=lambda s: (s <= 0.5).mean(),
             n='count')
        .reset_index()
        .sort_values('sentence_idx')
    )

    transitions = []
    positions = stats['sentence_idx'].tolist()

    for i in range(1, len(positions)):
        prev_idx = positions[i - 1]
        curr_idx = positions[i]

        # Only consider consecutive sentence steps
        if curr_idx != prev_idx + 1:
            continue

        prev_row = stats[stats['sentence_idx'] == prev_idx].iloc[0]
        curr_row = stats[stats['sentence_idx'] == curr_idx].iloc[0]

        mean_before = prev_row['mean_score']
        mean_after  = curr_row['mean_score']
        delta = mean_before - mean_after          # positive = drop, negative = rise

        comply_rate_prev = prev_row['comply_rate']
        refuse_rate_prev = prev_row['refuse_rate']
        comply_rate_curr = curr_row['comply_rate']
        refuse_rate_curr = curr_row['refuse_rate']

        # -- comply → refuse (score drops) ------------------------------------
        if (mean_before >= args.comply_threshold and
                mean_after  <= args.refuse_threshold and
                comply_rate_prev >= args.min_rate and
                refuse_rate_curr >= args.min_rate):

            direction = 'comply->refuse'
            rate_before = comply_rate_prev   # fraction complying before
            rate_after  = refuse_rate_curr   # fraction refusing after
            score_delta = round(delta, 4)

        # -- refuse → comply (score rises) ------------------------------------
        elif (mean_before <= args.refuse_threshold and
                mean_after  >= args.comply_threshold and
                refuse_rate_prev >= args.min_rate and
                comply_rate_curr >= args.min_rate):

            direction = 'refuse->comply'
            rate_before = refuse_rate_prev   # fraction refusing before
            rate_after  = comply_rate_curr   # fraction complying after
            score_delta = round(-delta, 4)   # store as positive magnitude

        else:
            continue

        # Extract sentence text using any repetition where both positions exist
        prev_threads = df[df['sentence_idx'] == prev_idx]['combined_thread'].dropna()
        curr_threads = df[df['sentence_idx'] == curr_idx]['combined_thread'].dropna()

        sentence_text = ''
        if not prev_threads.empty and not curr_threads.empty:
            sentence_text = extract_added_sentence(
                prev_threads.iloc[0], curr_threads.iloc[0]
            ).strip()

        transitions.append({
            'prompt_idx':   prompt_idx,
            'cot_idx':      cot_idx,
            'sentence_idx': curr_idx,
            'direction':    direction,
            'sentence_text': sentence_text,
            'mean_before':  round(mean_before, 4),
            'mean_after':   round(mean_after, 4),
            'score_delta':  score_delta,
            'rate_before':  round(rate_before, 4),
            'rate_after':   round(rate_after, 4),
            'std_before': round(
                prev_row['std_score'] if not np.isnan(prev_row['std_score']) else 0.0, 4),
            'std_after': round(
                curr_row['std_score'] if not np.isnan(curr_row['std_score']) else 0.0, 4),
            'n_reps_before': int(prev_row['n']),
            'n_reps_after':  int(curr_row['n']),
            'filepath': filepath,
        })

    return transitions


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _print_group(group_df: pd.DataFrame, direction: str, top_n: int = 10):
    label = 'comply → refuse' if direction == 'comply->refuse' else 'refuse → comply'
    print(f'\n  [{label}]  {len(group_df)} transition(s)')
    print(f'  {"─"*76}')

    ranked = group_df.sort_values(
        ['score_delta', 'rate_before'], ascending=[False, False]
    ).head(top_n).reset_index(drop=True)

    for _, row in ranked.iterrows():
        print(f'  prompt={row["prompt_idx"]}  cot={row["cot_idx"]}  '
              f'sentence_idx={row["sentence_idx"]}')
        print(f'    score:  {row["mean_before"]:.3f} → {row["mean_after"]:.3f}  '
              f'(|delta|={row["score_delta"]:.3f})')
        print(f'    rate before: {row["rate_before"]:.0%}   '
              f'rate after: {row["rate_after"]:.0%}')
        text = row['sentence_text']
        if len(text) > 200:
            text = text[:197] + '...'
        print(f'    sentence: {text!r}')
        print()


def _print_recurring(group_df: pd.DataFrame, direction: str):
    freq = (
        group_df.groupby('sentence_text')
        .agg(
            occurrences=('sentence_text', 'count'),
            mean_delta=('score_delta', 'mean'),
            mean_rate_before=('rate_before', 'mean'),
            mean_rate_after=('rate_after', 'mean'),
        )
        .sort_values(['occurrences', 'mean_delta'], ascending=[False, False])
        .head(10)
    )
    recurring = freq[freq['occurrences'] >= 2]
    if recurring.empty:
        return

    label = 'comply → refuse' if direction == 'comply->refuse' else 'refuse → comply'
    print(f'  TOP RECURRING [{label}] SENTENCES\n')
    for text, row in recurring.iterrows():
        display = text if len(text) <= 160 else text[:157] + '...'
        print(f'  [{int(row["occurrences"])}x]  |delta|={row["mean_delta"]:.3f}  '
              f'rate {row["mean_rate_before"]:.0%} → {row["mean_rate_after"]:.0%}')
        print(f'         {display!r}')
        print()


def print_summary(results_df: pd.DataFrame, top_n: int = 10):
    if results_df.empty:
        print('No transitions found with the given thresholds.')
        return

    print(f'\n{"="*80}')
    print(f'  HINGE-SENTENCE ANALYSIS  —  {len(results_df)} transition(s) detected')
    print(f'{"="*80}')

    for direction in ('comply->refuse', 'refuse->comply'):
        sub = results_df[results_df['direction'] == direction]
        if sub.empty:
            continue
        _print_group(sub, direction, top_n=top_n)

    # Recurring sentences per direction
    if 'sentence_text' in results_df.columns:
        print(f'{"─"*80}')
        for direction in ('comply->refuse', 'refuse->comply'):
            sub = results_df[results_df['direction'] == direction]
            if len(sub) > 1:
                _print_recurring(sub, direction)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    pattern = os.path.join(
        args.results_dir, args.model_name, 'dataset', 'quadrants',
        f'scored_full_resample_prompt*_cot*_rep_{args.repetitions}.csv'
    )
    input_paths = sorted(glob.glob(pattern))

    if not input_paths:
        print(f'No files found matching:\n  {pattern}')
        return

    print(f'Scanning {len(input_paths)} file(s)...')

    all_transitions = []
    for path in input_paths:
        transitions = analyse_file(path, args)
        all_transitions.extend(transitions)

    results_df = pd.DataFrame(all_transitions)

    if results_df.empty:
        print('No transitions found. Try lowering --comply_threshold, '
              '--refuse_threshold, or --min_rate.')
        return

    # Output CSV
    if args.output:
        out_path = args.output
    else:
        out_dir = os.path.join(args.results_dir, args.model_name, 'figures')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'hinge_sentences_rep{args.repetitions}.csv')

    save_cols = [c for c in results_df.columns if c != 'filepath']
    results_df[save_cols].sort_values(
        ['direction', 'score_delta', 'rate_before'], ascending=[True, False, False]
    ).to_csv(out_path, index=False)
    print(f'Results saved to: {out_path}')

    print_summary(results_df)


if __name__ == '__main__':
    main()
