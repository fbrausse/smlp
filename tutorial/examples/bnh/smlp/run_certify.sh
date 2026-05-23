#!/usr/bin/env bash
# Problem for function f₁(x) = 4x₁² + 4x₂²
# Certify stability of analytical Pareto solution for witness x₁ = x₂ = 0.294118 and given absolute radius rad-abs
# Expected result:
# From inequality: f₁(x) - 0.692042 < 2
# | x | < sqrt((2+0.692042)/8) = 0.580091
# Pass/Fail absolute radius boundary point: 0.580091-0.294118 = 0.285973
# Checking two values of rad-abs: 0.285 and 0.286
# Expected result: 0.285 -> PASS, 0.286 -> FAIL
# Reference: https://www.wolframalpha.com/input?i=8*x%5E2+-+0.692042+%3C+2

script_path="$(dirname "$(realpath "$0")")"
test=BNH
test_lc="$(echo "$test" | tr '[:upper:]' '[:lower:]')"
gzipped_dataset="${test_lc}.csv.gz"

if [[ $# -gt 0 && "$1" == "-clean" ]]; then
    rm -f ${test}* *.log  2>/dev/null
    rm -f "$gzipped_dataset"  2>/dev/null
    rm -f *.png 2>/dev/null
    exit 0
fi

result="${test}_certify.txt"
rm -f "$result" 2>/dev/null

tee_result() { tee -a "$result"; }

echo "=============================================== PROBLEM ======================================================" | tee_result
echo 'For function f₁(x₁,x₂) = 4x₁² + 4x₂²                                                                          ' | tee_result
echo 'Certify stability of analytical Pareto solution for witness f₁(x₁,x₂) = 0.692042, where  x₁' = x₂' = 0.294118 ' | tee_result
echo "And specified absolute radius rad-abs                                                                         " | tee_result
echo "----------------------------------------------- Expected result: ---------------------------------------------" | tee_result
echo 'From inequality: (f₁(x₁,x₂) - 0.692042)² < 4                                                                  ' | tee_result
echo '| x₁ | = | x₂ |  < sqrt((2+0.692042)/8) = 0.580091                                                            ' | tee_result
echo "Pass/Fail absolute radius boundary point: 0.580091-0.294118 = 0.285973                                        " | tee_result
echo "Checking two values of rad-abs: 0.285 and 0.286                                                               " | tee_result
echo 'Expected result: 0.285 -> PASS, 0.286 -> FAIL                                                                 ' | tee_result
echo "==============================================================================================================" | tee_result
echo "=============================================== SMLP CERTIFY SOLUTIONS =======================================" | tee_result

"${script_path}/bnh_dataset.py"

csv="${test}.csv"
json_pass="${script_path}/${test_lc}_certify_pass.json"
json_fail="${script_path}/${test_lc}_certify_fail.json"

gunzip -c "$gzipped_dataset" > "$csv"

e1="(F1-0.692042)*(F1-0.692042) < 4"

log_file="$(basename "$0").log"

for json in "$json_pass" "$json_fail"; do
    jq '[.variables[] | select(has("rad-abs")) | {label: .label, "rad-abs": .["rad-abs"]}]' "$json" | \
        tr -d '\n {}[]"' | sed -e 's/label://g' -e 's/,X2/; X2/' -e 's/:/=/g' -e 's/,/: /g' | tee -a "$result"
    echo "" | tee -a "$result"
    smlp \
        -data "$csv" \
        -spec "$json" \
        -out_dir ./ \
        -pref "$test" \
        -mode certify \
        -quer_names query1 \
        -quer_exprs "$e1" \
        -pareto f \
        -resp F1,F2 \
        -feat X1,X2 \
        -model poly_sklearn \
        -tree_encoding flat \
        -compress_rules t \
        -mrmr_pred 0 \
        -epsilon 0.000005 \
        -delta_rel 0.05 \
        -save_model t \
        -model_name "${csv%.*}_model" \
        -save_model_config t \
        -plots f \
        -pred_plots f \
        -resp_plots f \
        -seed 10 \
        -log_time f >> "$log_file" 2>&1
    echo "$e1 $(jq ".query1.witness_status" "${test}_${test}_certify_results.json")" | tee -a "$result"
done
echo "=============================================== DONE ========================================================" | tee_result
"${script_path}/witness_certify_plot.py" -timeout 5
