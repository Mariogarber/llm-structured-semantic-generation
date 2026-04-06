kubectl apply -f labeled_code.yaml

desc=$(kubectl describe limitrange limits-test)
max_cpu=$(echo "$desc" | egrep 'Container\s+cpu' | awk '{print $4}')
min_cpu=$(echo "$desc" | egrep 'Container\s+cpu' | awk '{print $3}')
max_mem=$(echo "$desc" | egrep 'Container\s+memory' | awk '{print $4}')
min_mem=$(echo "$desc" | egrep 'Container\s+memory' | awk '{print $3}')

if [ "$max_cpu" == "2" ] && [ "$min_cpu" == "100m" ] && [ "$max_mem" == "500Mi" ] && [ "$min_mem" == "100Mi" ]; then
    echo "Continue"
else
    echo "Test failed"
    exit 1
fi

MESSAGE=$(cat <<EOF | kubectl apply -f - 2>&1
apiVersion: v1
kind: Pod
metadata:
  name: test-limitrange
spec:
  containers:
  - name: test-container
    image: busybox
    command: ["/bin/sh"]
    args: ["-c", "sleep 3600"]
    resources:
      requests:
        cpu: "50m"
        memory: "50Mi"
EOF
)
sleep 5

if [[ $MESSAGE == *"minimum cpu usage per Container is 100m, but request is 50m"* ]]; then
    echo cloudeval_unit_test_passed
else
    kubectl delete pod test-limitrange
    echo "Test failed"
fi

