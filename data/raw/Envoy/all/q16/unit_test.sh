output=$(timeout -s SIGKILL 120 bash verify.sh 2>&1)
if [ $? -eq 137 ]; then
    echo "cloudeval_unit_test_timeout"
    exit 1
fi
docker compose down --remove-orphans --rmi all --volumes

echo "$output"

if ! echo "$output" | grep -q "ERROR:" && ! echo "$output" | grep -q "Wait.*failed" && echo "$output" | tail -n 1 | grep -q "Success"; then
    echo "cloudeval_unit_test_passed"
fi