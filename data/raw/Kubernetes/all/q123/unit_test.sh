kubectl apply -f labeled_code.yaml

ERROR_MESSAGE=$(cat <<EOF | kubectl apply -f - 2>&1
apiVersion: v1
kind: Pod
metadata:
  name: test-limitrange
spec:
  containers:
  - name: busybox
    image: busybox
    resources:
      requests:
        cpu: "170m"
        memory: "290Mi"
    command:
    - sleep
    - "3600"
EOF
)
sleep 5

if [[ $ERROR_MESSAGE == *"No limit is specified, maximum memory usage per Pod is 250Mi"* ]]; then
    echo cloudeval_unit_test_passed
else
    kubectl delete pod test-limitrange
    echo "Test failed"
fi