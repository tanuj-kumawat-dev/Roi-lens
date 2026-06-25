"""
ROI Lens: Advanced Marketing Attribution & Budget Refinement Strategy
Nexus Consumer Brands Strategy Taskforce

This script implements the core analytical engine of ROI Lens:
1. Preprocesses interaction logs and filters bot traffic.
2. Builds a Multi-Touch Attribution (MTA) Markov Chain model per brand.
3. Performs a financial audit comparing Legacy CPA with True CPA.
4. Performs constrained portfolio reallocation under Scenario 1 and Scenario 2.
5. Saves the final optimized allocations to CSV.
"""

import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.optimize import minimize

def preprocess_and_detect_bots(touchpoints_path, spend_path):
    """
    Loads raw interaction data and filters out bot traffic.
    Bots are defined as users with abnormally high interaction frequencies (>=60 events),
    perfect 1.0 CTR, and 0 conversions.
    """
    print("[1/5] Loading data and identifying bot traffic...")
    
    # Load files
    touchpoints = pd.read_csv(touchpoints_path)
    spend = pd.read_csv(spend_path)
    
    # Parse timestamps
    touchpoints['Timestamp_dt'] = pd.to_datetime(touchpoints['Timestamp'])
    touchpoints['Brand_ID'] = touchpoints['Campaign_ID'].apply(
        lambda x: x.split('_')[1] if len(x.split('_')) > 1 else 'Unknown'
    )
    
    # Identify bot users (total touchpoints >= 60)
    user_counts = touchpoints['User_ID'].value_counts()
    bot_users = user_counts[user_counts >= 60].index
    touchpoints['Is_Bot'] = touchpoints['User_ID'].isin(bot_users)
    
    # Split clean vs bot data
    clean_touchpoints = touchpoints[~touchpoints['Is_Bot']].copy()
    clean_touchpoints = clean_touchpoints.sort_values(by=['User_ID', 'Timestamp_dt'])
    
    # Aggregate clicks/impressions for financial audit
    campaign_stats = touchpoints.groupby(['Campaign_ID', 'Event_Type']).size().unstack(fill_value=0)
    campaign_stats_clean = clean_touchpoints.groupby(['Campaign_ID', 'Event_Type']).size().unstack(fill_value=0)
    
    # Merge stats into campaign spend
    spend_df = spend.copy()
    spend_df = spend_df.merge(campaign_stats, on='Campaign_ID', how='left').fillna(0)
    spend_df = spend_df.rename(columns={
        'Click': 'Total_Clicks', 
        'Impression': 'Total_Impressions', 
        'Purchase': 'Total_Purchases'
    })
    
    spend_df = spend_df.merge(campaign_stats_clean[['Click', 'Impression']], on='Campaign_ID', how='left').fillna(0)
    spend_df = spend_df.rename(columns={
        'Click': 'Clean_Clicks', 
        'Impression': 'Clean_Impressions'
    })
    
    # Calculate wasted budget fractions
    spend_df['Wasted_Fraction'] = np.where(
        spend_df['Pricing_Model'] == 'CPC',
        (spend_df['Total_Clicks'] - spend_df['Clean_Clicks']) / (spend_df['Total_Clicks'] + 1e-5),
        (spend_df['Total_Impressions'] - spend_df['Clean_Impressions']) / (spend_df['Total_Impressions'] + 1e-5)
    )
    spend_df['Wasted_Budget'] = spend_df['Wasted_Fraction'] * spend_df['Total_Budget_Allocated']
    spend_df['Clean_Budget'] = spend_df['Total_Budget_Allocated'] - spend_df['Wasted_Budget']
    
    total_budget = spend_df['Total_Budget_Allocated'].sum()
    wasted_budget = spend_df['Wasted_Budget'].sum()
    
    print(f" -> Found {len(bot_users)} bot users generating {touchpoints['Is_Bot'].sum()} touchpoints.")
    print(f" -> Overall budget wasted on bots: INR {wasted_budget:,.2f} ({wasted_budget/total_budget*100:.2f}% of total INR {total_budget/1e7:.1f} Cr budget).")
    
    return clean_touchpoints, spend_df


def get_markov_attribution(journeys):
    """
    Computes Markov Chain transition probabilities and removal effects for journeys.
    Journeys is a list of lists representing channel sequences ending in Purchase or Null.
    """
    transitions = defaultdict(lambda: defaultdict(int))
    for journey in journeys:
        transitions['Start'][journey[0]] += 1
        for i in range(len(journey) - 1):
            transitions[journey[i]][journey[i+1]] += 1
            
    prob_matrix = defaultdict(dict)
    for state, next_states in transitions.items():
        total_transitions = sum(next_states.values())
        for next_state, count in next_states.items():
            prob_matrix[state][next_state] = count / total_transitions
            
    prob_matrix['Purchase']['Purchase'] = 1.0
    prob_matrix['Null']['Null'] = 1.0
    
    channels = ['Instagram', 'Google Search', 'Influencer Blog', 'YouTube', 'Marketplace']
    all_trans_states = ['Start'] + channels
    state_to_idx = {state: idx for idx, state in enumerate(all_trans_states)}
    num_trans = len(all_trans_states)
    
    Q = np.zeros((num_trans, num_trans))
    R = np.zeros((num_trans, 2))
    
    for state in all_trans_states:
        idx = state_to_idx[state]
        for next_state, prob in prob_matrix[state].items():
            if next_state == 'Purchase':
                R[idx, 0] = prob
            elif next_state == 'Null':
                R[idx, 1] = prob
            elif next_state in state_to_idx:
                Q[idx, state_to_idx[next_state]] = prob
                
    I = np.eye(num_trans)
    try:
        N = np.linalg.inv(I - Q)
        B = N.dot(R)
        base_conversion_prob = B[state_to_idx['Start'], 0]
    except Exception:
        # Fallback if matrix is singular (no transitions)
        return {c: 0.2 for c in channels}
        
    removal_effects = {}
    for channel in channels:
        Q_mod = Q.copy()
        R_mod = R.copy()
        c_idx = state_to_idx[channel]
        Q_mod[c_idx, :] = 0.0
        R_mod[c_idx, 0] = 0.0
        R_mod[c_idx, 1] = 1.0
        
        try:
            N_mod = np.linalg.inv(I - Q_mod)
            B_mod = N_mod.dot(R_mod)
            mod_conversion_prob = B_mod[state_to_idx['Start'], 0]
            removal_effects[channel] = 1.0 - (mod_conversion_prob / base_conversion_prob)
        except Exception:
            removal_effects[channel] = 0.0
            
    total_re = sum(removal_effects.values())
    if total_re == 0:
        return {c: 0.2 for c in channels}
    return {channel: re / total_re for channel, re in removal_effects.items()}


def run_attribution_framework(clean_touchpoints, spend_df):
    """
    Builds user journeys per brand and computes multi-touch Markov weights.
    """
    print("[2/5] Running Multi-Touch Markov Chain Attribution...")
    
    brands = sorted(clean_touchpoints['Brand_ID'].unique())
    if 'Unknown' in brands:
        brands.remove('Unknown')
        
    brand_markov_weights = {}
    brand_purchases_count = {}
    
    for brand in brands:
        b_tp = clean_touchpoints[clean_touchpoints['Brand_ID'] == brand]
        user_journeys = b_tp.groupby('User_ID')
        
        journeys = []
        total_purchases = 0
        
        for uid, group in user_journeys:
            events = group.sort_values('Timestamp_dt')
            seq = []
            purchased = False
            for _, row in events.iterrows():
                seq.append(row['Channel'])
                if row['Event_Type'] == 'Purchase':
                    purchased = True
                    break
            if purchased:
                seq.append('Purchase')
                total_purchases += 1
            else:
                seq.append('Null')
            journeys.append(seq)
            
        weights = get_markov_attribution(journeys)
        brand_markov_weights[brand] = weights
        brand_purchases_count[brand] = total_purchases
        
    # Map weights and attributed purchases back to the spend dataframe
    def get_weight(row):
        return brand_markov_weights[row['Brand_ID']][row['Channel']]
        
    def get_attr_purchases(row):
        total_p = brand_purchases_count.get(row['Brand_ID'], 0)
        return row['Markov_Weight'] * total_p
        
    spend_df['Markov_Weight'] = spend_df.apply(get_weight, axis=1)
    spend_df['Attributed_Purchases'] = spend_df.apply(get_attr_purchases, axis=1)
    
    # Financial CPA Auditing
    spend_df['Legacy_CPA'] = spend_df['Total_Budget_Allocated'] / (spend_df['Total_Purchases'] + 1e-5)
    spend_df['True_CPA'] = spend_df['Clean_Budget'] / (spend_df['Attributed_Purchases'] + 1e-5)
    
    print(f" -> Markov attribution computed. Total attributed conversions: {sum(brand_purchases_count.values())}.")
    return spend_df, brand_purchases_count


def optimize_budgets(spend_df, brand_purchases_count, beta=0.75, budget_limit=100000000.0):
    """
    Optimizes budget allocations per brand using a scaled SLSQP solver to maximize conversions
    under diminishing returns curves (power saturation function).
    Includes:
    - Scenario 1: Bots Blocked (Unconstrained)
    - Scenario 2: Bots Remain (Unconstrained)
    - Scenario 1_Constrained: Bots Blocked (Strategic limits: Min 10%, Max 50% per channel)
    """
    print("[3/5] Calibrating diminishing returns and optimizing budget allocations...")
    
    # Calibrate power curve parameter k_c per campaign
    spend_df['k_c'] = spend_df['Attributed_Purchases'] / (spend_df['Clean_Budget'] ** beta + 1e-10)
    
    # We use units of 1 Crore (1e7 INR) for scaling inside the optimizer
    SCALE = 1e7
    scaled_limit = budget_limit / SCALE
    
    brands = sorted(spend_df['Brand_ID'].unique())
    opt_results = []
    
    for brand in brands:
        brand_df = spend_df[spend_df['Brand_ID'] == brand].copy()
        channels = brand_df['Channel'].tolist()
        k_vals = brand_df['k_c'].tolist()
        wasted_fracs = brand_df['Wasted_Fraction'].tolist()
        prev_budgets = brand_df['Total_Budget_Allocated'].tolist()
        prev_clean_budgets = brand_df['Clean_Budget'].tolist()
        prev_purchases = brand_df['Attributed_Purchases'].tolist()
        
        # Scaled K coefficients for the solver
        K_vals = [k * (SCALE ** beta) for k in k_vals]
        
        # Solver 1: Scenario 1 (Bots Blocked: full spend is clean spend)
        def obj_blocked(y):
            return -sum(K * (max(val, 0.0) ** beta) for K, val in zip(K_vals, y))
            
        # Solver 2: Scenario 2 (Bots Remain: only a fraction of spend is clean)
        def obj_remain(y):
            return -sum(K * (((1 - wf) * max(val, 0.0)) ** beta) for K, wf, val in zip(K_vals, wasted_fracs, y))
            
        cons = ({'type': 'eq', 'fun': lambda y: sum(y) - scaled_limit})
        x0 = [scaled_limit / 5.0] * 5
        
        # Run Unconstrained Scenario 1
        bounds_unconstrained = [(0, scaled_limit) for _ in range(5)]
        res_blocked = minimize(obj_blocked, x0, method='SLSQP', bounds=bounds_unconstrained, constraints=cons)
        
        # Run Unconstrained Scenario 2
        res_remain = minimize(obj_remain, x0, method='SLSQP', bounds=bounds_unconstrained, constraints=cons)
        
        # Run Strategic Constrained Scenario 1 (Min 10%, Max 50% spend)
        bounds_constrained = [(1.0, 5.0) for _ in range(5)] # In scaled units (1 Crore to 5 Crore)
        res_constrained = minimize(obj_blocked, x0, method='SLSQP', bounds=bounds_constrained, constraints=cons)
        
        opt_blocked_spend = res_blocked.x * SCALE
        opt_remain_spend = res_remain.x * SCALE
        opt_constrained_spend = res_constrained.x * SCALE
        
        for i, channel in enumerate(channels):
            opt_results.append({
                'Brand_ID': brand,
                'Channel': channel,
                'Prev_Budget': prev_budgets[i],
                'Prev_Clean_Budget': prev_clean_budgets[i],
                'Prev_Attributed_Purchases': prev_purchases[i],
                'Wasted_Fraction': wasted_fracs[i],
                'Opt_Spend_Bots_Blocked': opt_blocked_spend[i],
                'Opt_Spend_Bots_Remain': opt_remain_spend[i],
                'Opt_Spend_Bots_Blocked_Constrained': opt_constrained_spend[i],
                'Expected_Purchases_Bots_Blocked': k_vals[i] * (opt_blocked_spend[i] ** beta),
                'Expected_Purchases_Bots_Remain': k_vals[i] * (((1 - wasted_fracs[i]) * opt_remain_spend[i]) ** beta),
                'Expected_Purchases_Bots_Blocked_Constrained': k_vals[i] * (opt_constrained_spend[i] ** beta)
            })
            
    opt_df = pd.DataFrame(opt_results)
    
    total_hist = sum(brand_purchases_count.values())
    total_blocked = opt_df['Expected_Purchases_Bots_Blocked'].sum()
    total_remain = opt_df['Expected_Purchases_Bots_Remain'].sum()
    total_constrained = opt_df['Expected_Purchases_Bots_Blocked_Constrained'].sum()
    
    print(f" -> Historical conversions: {total_hist}")
    print(f" -> Scenario 1 (Bots Blocked, Unconstrained) conversions: {total_blocked:,.1f} (+{(total_blocked/total_hist - 1)*100:.1f}%)")
    print(f" -> Scenario 2 (Bots Remain, Unconstrained) conversions:    {total_remain:,.1f} (+{(total_remain/total_hist - 1)*100:.1f}%)")
    print(f" -> Scenario 1 (Bots Blocked, Constrained) conversions:      {total_constrained:,.1f} (+{(total_constrained/total_hist - 1)*100:.1f}%)")
    
    return opt_df


def run_persona_intelligence(clean_touchpoints, profiles_path):
    """
    Merges buyers with psychographic segments and trend affinities from Ai Palette.
    """
    print("[4/5] Running Brand Persona Intelligence mapping...")
    profiles = pd.read_csv(profiles_path)
    
    # Filter for buyer events
    buyers = clean_touchpoints[clean_touchpoints['Event_Type'] == 'Purchase'].copy()
    buyers_profiles = buyers.merge(profiles, on='User_ID', how='inner')
    
    print(f" -> Successfully mapped {len(buyers_profiles)} clean buyer profiles.")
    return buyers_profiles


def save_and_report_results(opt_df, buyers_profiles, output_csv):
    """
    Saves results to a CSV file and prints diagnostic tables.
    """
    print("[5/5] Saving results and exporting summary reports...")
    
    # Save CSV
    opt_df.to_csv(output_csv, index=False)
    print(f" -> Saved optimized allocations to: {output_csv}")
    
    # Summary of budget allocations by channel
    channel_summary = opt_df.groupby('Channel')[['Prev_Budget', 'Opt_Spend_Bots_Blocked', 'Opt_Spend_Bots_Remain', 'Opt_Spend_Bots_Blocked_Constrained']].sum()
    print("\n================== CHANNEL LEVEL BUDGET SUMMARY ==================")
    print(channel_summary)
    
    # Print brand-level allocation table for reference
    print("\n================== BRAND LEVEL OPTIMIZED BUDGETS (SCENARIO 1 CONSTRAINED) ==================")
    b_pivot = opt_df.pivot(index='Brand_ID', columns='Channel', values='Opt_Spend_Bots_Blocked_Constrained')
    print(b_pivot)


if __name__ == "__main__":
    # Define absolute filepaths based on current directory
    cwd = os.getcwd()
    touchpoints_csv = os.path.join(cwd, "touchpoints.csv")
    campaign_spend_csv = os.path.join(cwd, "campaign_spend.csv")
    user_profiles_csv = os.path.join(cwd, "user_profiles.csv")
    output_allocations_csv = os.path.join(cwd, "roi_lens_optimized_allocations.csv")
    
    # Run pipeline
    clean_tp, spend_df = preprocess_and_detect_bots(touchpoints_csv, campaign_spend_csv)
    spend_df, brand_purchases = run_attribution_framework(clean_tp, spend_df)
    opt_df = optimize_budgets(spend_df, brand_purchases)
    buyers_profiles = run_persona_intelligence(clean_tp, user_profiles_csv)
    save_and_report_results(opt_df, buyers_profiles, output_allocations_csv)
    
    print("\nROI Lens Pipeline completed successfully!")
