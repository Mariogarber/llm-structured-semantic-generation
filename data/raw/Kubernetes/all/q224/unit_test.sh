kubectl create namespace test
kubectl create namespace test3
kubectl create serviceaccount myaccount -n test

kubectl apply -f labeled_code.yaml

cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: test-serviceaccount-pod
  namespace: test
spec:
  serviceAccountName: myaccount
  containers:
  - name: test-container
    image: busybox
    command: ["/bin/sh"]
    args: ["-c", "sleep 3600"]
EOF

cleanup() {
  kubectl delete pod test-serviceaccount-pod -n test
  kubectl delete namespace test
  kubectl delete namespace test3
  kubectl delete -f labeled_code.yaml
}

kubectl wait --for=condition=Ready pod/test-serviceaccount-pod -n test --timeout=60s

if kubectl auth can-i list pods -n test3 --as=system:serviceaccount:test:myaccount; then
  echo cloudeval_unit_test_passed
else
  echo "Test failed"
fi

cleanup
