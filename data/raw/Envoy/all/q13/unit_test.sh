output=$(bash verify.sh 2>&1)

echo "$output"

if ! echo "$output" | grep "ERROR:"; then
    if echo "$output" | tail -n 1 | grep -q "Success"; then
        echo "cloudeval_unit_test_passed"
    fi
fi
# [Envoy docs](https://www.envoyproxy.io/docs/envoy/latest/start/sandboxes/redis.html).