kubectl apply -f labeled_code.yaml

cat <<EOF | kubectl apply -f -
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: restricted
spec:
  privileged: false
  allowPrivilegeEscalation: false
EOF

kubectl create serviceaccount test-sa

cat <<EOF | kubectl apply -f -
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: test-crb
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: restricted-psp-user
subjects:
- kind: ServiceAccount
  name: test-sa
  namespace: default
EOF

cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: test-pod
spec:
  serviceAccountName: test-sa
  containers:
  - name: test-container
    image: busybox
    command: ["/bin/sh", "-c", "sleep 3600"]
EOF

kubectl wait --for=condition=Ready pod/test-pod --timeout=60s

desc=$(kubectl describe pod test-pod)
if [[ $desc == *"test-sa"* ]]; then
    if kubectl get clusterrole restricted-psp-user | grep "CREATED AT"; then
        echo cloudeval_unit_test_passedß
    fi
else
    echo "Test failed"
fi

kubectl delete pod test-pod
kubectl delete serviceaccount test-sa
kubectl delete clusterrolebinding test-crb
kubectl delete podsecuritypolicy restricted
kubectl delete -f labeled_code.yaml
