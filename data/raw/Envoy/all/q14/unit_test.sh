# This unit test cannot work with timeout
bash verify.sh > output.txt 2>&1

PID=$!
timeout 120 bash -c "while ps -p $PID > /dev/null; do sleep 1; done"
timeout_status=$?
kill $PID
docker compose down --remove-orphans --rmi all --volumes

if [ $timeout_status -eq 124 ]; then
    echo "cloudeval_unit_test_timeout"
    exit 1
fi

output=$(cat output.txt)
rm output.txt
echo "$output"

if ! echo "$output" | grep -q "ERROR:" && ! echo "$output" | grep -q "Wait.*failed" && echo "$output" | tail -n 1 | grep -q "Success"; then
    echo "cloudeval_unit_test_passed"
fi