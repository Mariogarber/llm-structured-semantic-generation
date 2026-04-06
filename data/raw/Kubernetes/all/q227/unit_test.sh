kubectl create namespace test2
kubectl create serviceaccount myaccount --namespace=test2
sleep 5

cat <<EOF | kubectl apply -f -
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: test2
  name: testadmin
rules:
- apiGroups: ["*"]
  resources: ["*"]
  verbs: ["*"]
EOF

sleep 5

kubectl apply -f labeled_code.yaml

cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: test-role-pod
  namespace: test2
spec:
  serviceAccountName: myaccount
  containers:
  - name: test-container
    image: bitnami/kubectl
    command: ["/bin/sh", "-c", "sleep 3600"]
EOF

cleanup() {
  kubectl delete pod test-role-pod --namespace=test2
  kubectl delete serviceaccount myaccount -n test2
  kubectl delete role testadmin --namespace=test2
  kubectl delete namespace test2
}

kubectl wait --for=condition=Ready pod/test-role-pod --namespace=test2 --timeout=60s

if kubectl auth can-i list pods --as=system:serviceaccount:test2:myaccount -n test2; then
    echo cloudeval_unit_test_passed
else
    echo "Test failed"
fi

cleanup