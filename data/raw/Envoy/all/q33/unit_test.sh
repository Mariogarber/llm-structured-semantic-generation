output=$(bash verify.sh)

echo "$output"

if echo "$output" | tail -n 1 | grep -q "Success"; then
    echo "cloudeval_unit_test_passed"
fi