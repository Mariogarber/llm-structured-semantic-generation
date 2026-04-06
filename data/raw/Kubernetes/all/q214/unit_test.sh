kubectl apply -f labeled_code.yaml

cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ServiceAccount
metadata:
  name: manager-service-account
EOF

cat <<EOF | kubectl apply -f -
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: manager-group-assign
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: secret-reader
subjects:
- kind: ServiceAccount
  name: manager-service-account
  namespace: default
EOF

kubectl create secret generic secret-to-read --from-literal=password=supersecret

cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: secret-reader-pod
spec:
  serviceAccountName: manager-service-account
  containers:
  - name: secret-reader-container
    image: busybox
    command: ["/bin/sh", "-c", "sleep 3600"]
EOF

cleanup() {
  kubectl delete pod secret-reader-pod
  kubectl delete serviceaccount manager-service-account
  kubectl delete rolebinding manager-group-assign
  kubectl delete secret secret-to-read
  kubectl delete -f labeled_code.yaml
}

kubectl wait --for=condition=Ready pod/secret-reader-pod --timeout=60s

if kubectl exec secret-reader-pod -- sh -c "cat /var/run/secrets/kubernetes.io/serviceaccount/token" > /dev/null; then
  if kubectl get clusterrolebinding read-secrets-global | grep "ROLE"; then
    echo cloudeval_unit_test_passed
  fi
else
  echo "Test failed"
fi

cleanup