kubectl apply -f labeled_code.yaml

cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: test-limitrange
  labels:
    purpose: test-limitrange
spec:
  containers:
  - name: test-container
    image: busybox
    command: ["/bin/sh"]
    args: ["-c", "sleep 3600"]
EOF

kubectl wait --for=condition=Ready pod/test-limitrange --timeout=60s

kubectl describe pod test-limitrange | egrep "memory:\s+256Mi" && echo cloudeval_unit_test_passed

kubectl delete pod test-limitrange
