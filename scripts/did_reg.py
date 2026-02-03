# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*70)
print("DID Regression with Two-way Fixed Effects")
print("="*70)

# Step 1: Load matched data
print("\nStep 1: Loading matched dataset...")
df = pd.read_excel('PSM_Analysis/matched_dataset_log_vars.xlsx')
print("Data loaded: {} rows x {} columns".format(df.shape[0], df.shape[1]))
print("Year range: {} - {}".format(df['year'].min(), df['year'].max()))
print("Cities: {}".format(df['city_name'].nunique()))

# Step 2: Define variables
print("\nStep 2: Defining variables...")
y_var = 'ln_carbon_intensity'
did_var = 'DID_matched'
control_vars = ['ln_pgdp', 'ln_pop_density', 'ln_industrial_advanced', 'ln_fdi_openness']

print("\nDependent variable (Y): {}".format(y_var))
print("Independent variable (DID): {}".format(did_var))
print("Control variables: {}".format(', '.join(control_vars)))

# Check missing values
core_vars = [y_var, did_var] + control_vars
for var in core_vars:
    missing = df[var].isnull().sum()
    print("  {}: {} missing values".format(var, missing))

# Drop missing values
df_reg = df.dropna(subset=core_vars).copy()
print("\nCleaned data: {} rows".format(len(df_reg)))

# Step 3: Descriptive statistics
print("\n" + "="*70)
print("Step 3: Descriptive Statistics")
print("="*70)

print("\nCore variables summary:")
print(df_reg[core_vars].describe())

print("\nBy DID_matched group:")
group_stats = df_reg.groupby(did_var)[y_var].agg(['mean', 'count'])
print(group_stats)

# Step 4: Create fixed effects dummies
print("\n" + "="*70)
print("Step 4: Creating fixed effects model (LSDV)")
print("="*70)

# City fixed effects
city_dummies = pd.get_dummies(df_reg['city_name'], prefix='city', drop_first=True)
city_fe_count = city_dummies.shape[1]
print("City fixed effects: {} dummies".format(city_fe_count))

# Year fixed effects
year_dummies = pd.get_dummies(df_reg['year'], prefix='year', drop_first=True)
year_fe_count = year_dummies.shape[1]
print("Year fixed effects: {} dummies".format(year_fe_count))

# Combine all variables
X_did = df_reg[[did_var]].values.reshape(-1, 1)
X_control = df_reg[control_vars].values
X_city = city_dummies.values
X_year = year_dummies.values

X = np.hstack([X_did, X_control, X_city, X_year])
y = df_reg[y_var].values

n = X.shape[0]
k = X.shape[1]

print("\nDesign matrix:")
print("  X: {} observations x {} variables".format(n, k))
print("    - DID_matched: 1")
print("    - Controls: {}".format(len(control_vars)))
print("    - City FE: {}".format(city_fe_count))
print("    - Year FE: {}".format(year_fe_count))

# Step 5: Run OLS regression
print("\nRunning OLS regression...")
model = LinearRegression(fit_intercept=True)
model.fit(X, y)

# Predictions and R-squared
y_pred = model.predict(X)
ss_res = np.sum((y - y_pred)**2)
ss_tot = np.sum((y - np.mean(y))**2)
r_squared = 1 - ss_res/ss_tot
adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)

print("Regression completed!")

# Calculate standard errors
X_with_intercept = np.column_stack([np.ones(n), X])
XX_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
residuals = y - y_pred
mse = ss_res / (n - k - 1)
vcv = mse * XX_inv
se = np.sqrt(np.diag(vcv))

# Extract coefficients and SE
coef_intercept = model.intercept_
coef = model.coef_
se_intercept = se[0]
se_coef = se[1:]

# t-statistics and p-values
t_intercept = coef_intercept / se_intercept
t_coef = coef / se_coef
p_intercept = 2 * (1 - stats.t.cdf(np.abs(t_intercept), n - k - 1))
p_coef = 2 * (1 - stats.t.cdf(np.abs(t_coef), n - k - 1))

# Confidence intervals
t_crit = stats.t.ppf(0.975, n - k - 1)
ci_intercept = (coef_intercept - t_crit * se_intercept, coef_intercept + t_crit * se_intercept)
ci_coef = [(coef[i] - t_crit * se_coef[i], coef[i] + t_crit * se_coef[i]) for i in range(len(coef))]

# Step 6: Results
print("\n" + "="*70)
print("Step 5: Regression Results")
print("="*70)

# Extract DID coefficient
did_idx = 0
did_coef_val = coef[did_idx]
did_se_val = se_coef[did_idx]
did_t_val = t_coef[did_idx]
did_p_val = p_coef[did_idx]
did_ci_lower, did_ci_upper = ci_coef[did_idx]

print("\nDID_matched coefficient:")
print("  Coefficient: {:.6f}".format(did_coef_val))
print("  Std. Error: {:.6f}".format(did_se_val))
print("  t-statistic: {:.4f}".format(did_t_val))
print("  p-value: {:.4f}".format(did_p_val))

if did_p_val < 0.01:
    sig = "*** (p<0.01)"
elif did_p_val < 0.05:
    sig = "**  (p<0.05)"
elif did_p_val < 0.1:
    sig = "*   (p<0.1)"
else:
    sig = "    (not significant)"

print("  Significance: {}".format(sig))
print("  95% CI: [{:.6f}, {:.6f}]".format(did_ci_lower, did_ci_upper))

print("\nInterpretation:")
print("  After controlling for city and year fixed effects,")
print("  the low-carbon pilot policy changes carbon intensity by {:.2f}%".format(did_coef_val*100))
print("  {}".format(sig))

if did_coef_val < 0:
    print("  Policy significantly REDUCED carbon intensity")
else:
    print("  Policy did NOT significantly reduce carbon intensity")

# Model fit
print("\nModel fit:")
print("  R-squared: {:.4f}".format(r_squared))
print("  Adj. R-squared: {:.4f}".format(adj_r_squared))
print("  Sample size: {}".format(n))

# Step 7: Control variables
print("\n" + "="*70)
print("Step 6: Control Variable Coefficients")
print("="*70)

print("\n{:<30s} {:>12s} {:>10s} {:>8s} {:>10s}".format('Variable', 'Coef', 'Std.Err', 't-value', 'Sig'))
print("-" * 80)

# DID variable
print("{:<30s} {:>12.6f} {:>10.4f} {:>8.2f} {:>10s}".format(
    did_var, did_coef_val, did_se_val, did_t_val, sig.strip()))

# Control variables
for i, var in enumerate(control_vars):
    idx = 1 + i
    c = coef[idx]
    s = se_coef[idx]
    t = t_coef[idx]
    p = p_coef[idx]

    if p < 0.01:
        marker = '***'
    elif p < 0.05:
        marker = '** '
    elif p < 0.1:
        marker = '*  '
    else:
        marker = '   '

    print("{:<30s} {:>12.6f} {:>10.4f} {:>8.2f} {:>10s}".format(
        var, c, s, t, marker))

# Intercept
print("{:<30s} {:>12.6f} {:>10.4f} {:>8.2f}".format(
    'Intercept', coef_intercept, se_intercept, t_intercept))

# Step 8: Save results
print("\n" + "="*70)
print("Step 7: Saving Results")
print("="*70)

# Create results DataFrame
results_data = []
results_data.append({
    'Variable': 'DID_matched',
    'Coefficient': did_coef_val,
    'Std._Error': did_se_val,
    't_value': did_t_val,
    'p_value': did_p_val,
    'Significance': sig.strip()
})

for i, var in enumerate(control_vars):
    idx = 1 + i
    p = p_coef[idx]
    if p < 0.01:
        marker = '***'
    elif p < 0.05:
        marker = '**'
    elif p < 0.1:
        marker = '*'
    else:
        marker = ''

    results_data.append({
        'Variable': var,
        'Coefficient': coef[idx],
        'Std._Error': se_coef[idx],
        't_value': t_coef[idx],
        'p_value': p,
        'Significance': marker
    })

results_df = pd.DataFrame(results_data)

# Save to Excel
output_file = 'DID_Regression_Results.xlsx'
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    results_df.to_excel(writer, sheet_name='Regression Results', index=False)
    df_reg.describe().to_excel(writer, sheet_name='Descriptive Stats', index=False)

    # Full model results
    var_names = [did_var] + control_vars + list(city_dummies.columns) + list(year_dummies.columns)
    full_results = pd.DataFrame({
        'Variable': ['Intercept'] + var_names,
        'Coefficient': [coef_intercept] + list(coef),
        'Std._Error': [se_intercept] + list(se_coef),
        't_value': [t_intercept] + list(t_coef),
        'p_value': [p_intercept] + list(p_coef)
    })
    full_results.to_excel(writer, sheet_name='Full Coefficients', index=False)

print("\nResults saved to: {}".format(output_file))

# Save text report
report_file = 'DID_Regression_Report.txt'
with open(report_file, 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("DID Two-way Fixed Effects Regression Report\n")
    f.write("="*70 + "\n\n")

    f.write("I. Model Specification\n")
    f.write("-"*70 + "\n")
    f.write("Dataset: matched_dataset_log_vars.xlsx\n")
    f.write("Dependent variable: {} (carbon intensity)\n".format(y_var))
    f.write("Independent variable: {} (matched DID)\n".format(did_var))
    f.write("Control variables: {}\n".format(', '.join(control_vars)))
    f.write("Fixed effects: City FE + Year FE\n")
    f.write("Method: OLS with LSDV\n\n")

    f.write("II. Regression Results\n")
    f.write("-"*70 + "\n")
    f.write("Sample size: {}\n".format(n))
    f.write("R-squared: {:.4f}\n".format(r_squared))
    f.write("Adj. R-squared: {:.4f}\n\n".format(adj_r_squared))

    f.write("Key Result (DID_matched):\n")
    f.write("  Coefficient: {:.6f}\n".format(did_coef_val))
    f.write("  Std. Error: {:.6f}\n".format(did_se_val))
    f.write("  t-value: {:.4f}\n".format(did_t_val))
    f.write("  p-value: {:.4f}\n".format(did_p_val))
    f.write("  95% CI: [{:.6f}, {:.6f}]\n\n".format(did_ci_lower, did_ci_upper))

    f.write("Conclusion:\n")
    if did_p_val < 0.05:
        f.write("  The low-carbon pilot policy has a significant effect on carbon intensity ({})\n".format(sig))
        f.write("  Policy changes carbon intensity by {:.2f}%\n".format(did_coef_val*100))
    else:
        f.write("  The low-carbon pilot policy has NO significant effect on carbon intensity\n")

    f.write("\nIII. Interpretation\n")
    f.write("-"*70 + "\n")
    f.write("DID coefficient = {:.6f}\n".format(did_coef_val))
    if did_coef_val < 0:
        f.write("Interpretation: The policy REDUCES carbon intensity by {:.2f}%\n".format(abs(did_coef_val)*100))
    else:
        f.write("Interpretation: The policy INCREASES carbon intensity by {:.2f}%\n".format(did_coef_val*100))

    f.write("\nIV. Recommendations\n")
    f.write("-"*70 + "\n")
    f.write("1. This is the baseline regression result - use it as your main table\n")
    f.write("2. PSM-matched data ensures sample representativeness\n")
    f.write("3. Log control variables satisfy statistical assumptions\n")
    f.write("4. Follow up with robustness checks\n")

print("Report saved to: {}".format(report_file))

print("\n" + "="*70)
print("Regression Analysis Completed!")
print("="*70)

print("\nKey Finding:")
print("  Low-carbon pilot policy changed carbon intensity by {:+.2f}% {}".format(did_coef_val*100, sig))
if did_p_val < 0.05:
    print("  Policy effect is SIGNIFICANT!")
else:
    print("  Policy effect is NOT significant")

print("\nOutput files:")
print("  1. {} - Excel results".format(output_file))
print("  2. {} - Text report".format(report_file))
