kubectl apply -f labeled_code.yaml

cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: storage-test-pod
spec:
  containers:
  - name: test-container
    image: busybox
    command: ["/bin/sh", "-c", "sleep 3600"]
    resources:
      requests:
        ephemeral-storage: "2Gi"
EOF

sleep 10

ephemeral_storage_value=$(kubectl describe pod storage-test-pod | grep "ephemeral-storage:")
echo $ephemeral_storage_value

if [[ $(kubectl get limitrange | grep "storage-range-limits") ]] && [[ $ephemeral_storage_value == *2Gi* ]]; then
    echo "cloudeval_unit_test_passed"
else
    echo "Test failed"
fi


kubectl delete pod storage-test-pod
